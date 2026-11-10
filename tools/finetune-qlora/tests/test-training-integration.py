#!/usr/bin/env python3
"""Opt-in model-backed QLoRA training and checkpoint-resume integration test."""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


TRAIN_TEXTS = (
    (
        "Apples are often red or green. Oranges are orange and bananas are yellow. "
        "Fruit grows on trees and provides useful food for people. Apples can be sweet, "
        "while lemons taste sour. A small language model predicts the next token from "
        "earlier words in its context. The model should process the same input consistently "
        "when the dataset, seed, and training settings stay fixed. "
    ) * 10,
    (
        "A second training example describes the weather. Warm sunlight dries wet leaves, "
        "clouds can bring rain, and a strong wind moves branches. Simple sentences provide "
        "a separate sample so the integration test exercises more than one training window. "
    ) * 10,
)

EVAL_TEXT = " ".join(
    [
        "Small language models predict the next token from a sequence of earlier tokens.",
        "Training changes a small set of adapter weights while the base model remains frozen.",
        "A fixed evaluation corpus makes it possible to compare the model before and after an update.",
        "The purpose of this test is to verify that QLoRA training completes and the saved adapter loads for inference.",
    ]
    * 8
)


def print_subprocess_output(output, *, verbose, failed):
    if verbose:
        if output:
            print(output, flush=True)
        return
    if not failed:
        return

    lines = output.splitlines()
    keep = [
        line for line in lines
        if re.search(r"^\[FAIL\]|\b(?:error|failed|failure)\b", line, re.IGNORECASE)
    ]
    if not keep:
        keep = lines[-12:]
    for line in keep:
        print(line, flush=True)


def run(command, timeout, description, *, verbose=False):
    if verbose:
        print(f"[RUN] {description}: {' '.join(str(part) for part in command)}", flush=True)
    try:
        result = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print_subprocess_output(output, verbose=verbose, failed=True)
        raise RuntimeError(f"{description} timed out after {timeout} seconds") from exc

    print_subprocess_output(
        result.stdout or "", verbose=verbose, failed=result.returncode != 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{description} exited with status {result.returncode}")
    return result.stdout


def require_gguf(path, description):
    if not path.is_file():
        raise RuntimeError(f"{description} was not created: {path}")
    if path.stat().st_size < 8:
        raise RuntimeError(f"{description} is unexpectedly small: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError(f"{description} does not have GGUF magic: {path}")


def load_gguf_reader(path):
    repo_root = Path(__file__).resolve().parents[3]
    gguf_python = repo_root / "gguf-py"
    if str(gguf_python) not in sys.path:
        sys.path.insert(0, str(gguf_python))
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError("NumPy and the bundled gguf-py package are required to diagnose GGUF differences") from error
    return GGUFReader(str(path))


def gguf_tensor_as_float32(tensor):
    import numpy as np

    tensor_type = int(tensor.tensor_type)
    if tensor_type == 0:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    if tensor_type == 1:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(-1)
    if tensor_type == 30:
        raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)
        if raw.size % 2 != 0:
            raise RuntimeError(f"invalid BF16 tensor byte count for {tensor.name}")
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    raise RuntimeError(f"unsupported GGUF tensor type {tensor_type} for numerical comparison of {tensor.name}")


def compare_gguf_tensor_values(left_path, right_path):
    import numpy as np

    left_reader = load_gguf_reader(left_path)
    right_reader = load_gguf_reader(right_path)
    left = {tensor.name: tensor for tensor in left_reader.tensors}
    right = {tensor.name: tensor for tensor in right_reader.tensors}
    missing_left = sorted(right.keys() - left.keys())
    missing_right = sorted(left.keys() - right.keys())
    if missing_left or missing_right:
        return {
            "tensor_count_left": len(left),
            "tensor_count_right": len(right),
            "missing_left": missing_left[:8],
            "missing_right": missing_right[:8],
            "different_tensors": [],
        }

    different = []
    for name in sorted(left):
        left_tensor = left[name]
        right_tensor = right[name]
        if tuple(left_tensor.shape.tolist()) != tuple(right_tensor.shape.tolist()):
            different.append({"name": name, "shape_mismatch": True})
            continue
        left_values = gguf_tensor_as_float32(left_tensor)
        right_values = gguf_tensor_as_float32(right_tensor)
        if left_values.size != right_values.size:
            different.append({"name": name, "element_count_mismatch": True})
            continue
        delta = np.abs(left_values - right_values)
        max_abs = float(delta.max(initial=0.0))
        if max_abs == 0.0:
            continue
        rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        reference_max = float(np.abs(left_values).max(initial=0.0))
        different.append({
            "name": name,
            "elements": int(left_values.size),
            "max_abs": max_abs,
            "rms": rms,
            "reference_max_abs": reference_max,
            "relative_to_reference_max": max_abs / max(reference_max, np.finfo(np.float32).tiny),
        })

    different.sort(key=lambda item: item.get("max_abs", float("inf")), reverse=True)
    return {
        "tensor_count_left": len(left),
        "tensor_count_right": len(right),
        "different_tensors": different,
    }


def read_optimizer_state_blob(path):
    import numpy as np

    reader = load_gguf_reader(path)
    for tensor in reader.tensors:
        if tensor.name == "optimizer.state":
            return np.asarray(tensor.data, dtype=np.int8).tobytes()
    raise RuntimeError(f"optimizer.state tensor is missing from {path}")


def parse_optimizer_state(blob):
    offset = 0

    def read(fmt):
        nonlocal offset
        size = struct.calcsize(fmt)
        if offset + size > len(blob):
            raise RuntimeError("optimizer state is truncated")
        values = struct.unpack_from(fmt, blob, offset)
        offset += size
        return values if len(values) > 1 else values[0]

    if blob[:8] != b"GGOPTST1":
        raise RuntimeError("optimizer state has an unexpected magic value")
    offset = 8
    version = read("<I")
    optimizer = read("<I")
    state_type = read("<I") if version >= 2 else 0
    iteration = read("<Q")
    rng_size = read("<Q")
    if offset + rng_size > len(blob):
        raise RuntimeError("optimizer RNG state is truncated")
    rng_state = blob[offset:offset + rng_size]
    offset += rng_size
    tensor_count = read("<I")
    tensors = {}
    for _ in range(tensor_count):
        name_size = read("<I")
        if offset + name_size > len(blob):
            raise RuntimeError("optimizer tensor name is truncated")
        name = blob[offset:offset + name_size].decode("utf-8")
        offset += name_size
        tensor_type = read("<I")
        n_dims = read("<I")
        dimensions = read("<" + "Q" * 4)
        nbytes = read("<Q")
        if offset + nbytes > len(blob):
            raise RuntimeError(f"optimizer tensor data is truncated: {name}")
        tensors[name] = (tensor_type, dimensions, blob[offset:offset + nbytes])
        offset += nbytes
    if offset != len(blob):
        raise RuntimeError("optimizer state contains trailing data")
    return (version, optimizer, state_type, iteration, rng_state), tensors


def compare_optimizer_state_values(left_path, right_path):
    import numpy as np

    left_header, left_tensors = parse_optimizer_state(read_optimizer_state_blob(left_path))
    right_header, right_tensors = parse_optimizer_state(read_optimizer_state_blob(right_path))
    if left_header != right_header:
        return {"header_equal": False, "left_header": left_header[:4], "right_header": right_header[:4]}
    missing_left = sorted(right_tensors.keys() - left_tensors.keys())
    missing_right = sorted(left_tensors.keys() - right_tensors.keys())
    if missing_left or missing_right:
        return {"header_equal": True, "missing_left": missing_left[:8], "missing_right": missing_right[:8]}

    different = []
    dtype_by_ggml_type = {0: np.float32, 1: np.float16}
    for name in sorted(left_tensors):
        left_type, left_shape, left_data = left_tensors[name]
        right_type, right_shape, right_data = right_tensors[name]
        if left_type != right_type or left_shape != right_shape or len(left_data) != len(right_data):
            different.append({"name": name, "metadata_mismatch": True})
            continue
        dtype = dtype_by_ggml_type.get(left_type)
        if dtype is None:
            if left_data != right_data:
                different.append({"name": name, "non_float_data_differs": True})
            continue
        left_values = np.frombuffer(left_data, dtype=dtype).astype(np.float32)
        right_values = np.frombuffer(right_data, dtype=dtype).astype(np.float32)
        delta = np.abs(left_values - right_values)
        max_abs = float(delta.max(initial=0.0))
        if max_abs == 0.0:
            continue
        rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        reference_max = float(np.abs(left_values).max(initial=0.0))
        different.append({
            "name": name,
            "elements": int(left_values.size),
            "max_abs": max_abs,
            "rms": rms,
            "reference_max_abs": reference_max,
            "relative_to_reference_max": max_abs / max(reference_max, np.finfo(np.float32).tiny),
        })
    different.sort(key=lambda item: item.get("max_abs", float("inf")), reverse=True)
    return {"header_equal": True, "different_tensors": different}


def format_tensor_differences(summary, limit=8):
    if "different_tensors" not in summary:
        return str(summary)
    if summary.get("missing_left") or summary.get("missing_right"):
        return json.dumps(summary, sort_keys=True)
    if summary.get("tensor_count_left") != summary.get("tensor_count_right"):
        return json.dumps(summary, sort_keys=True)
    different = summary["different_tensors"]
    if not different:
        return "all tensor values match; the GGUF container bytes differ"
    return "; ".join(
        f"{item['name']}: max_abs={item.get('max_abs', 'n/a')}, "
        f"rms={item.get('rms', 'n/a')}, rel_max={item.get('relative_to_reference_max', 'n/a')}"
        for item in different[:limit]
    )


def require_cuda_offload(log, description):
    clean_log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)

    cuda_buffer_sizes = [
        float(size)
        for size in re.findall(
            r"\bCUDA\d+\b.*?\bmodel buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
            clean_log,
            re.IGNORECASE,
        )
    ]
    device_counts = [
        int(count)
        for count in re.findall(
            r"\bfound\s+(\d+)\s+CUDA devices\b",
            clean_log,
            re.IGNORECASE,
        )
    ]
    offloaded_layers = [
        (int(offloaded), int(total))
        for offloaded, total in re.findall(
            r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU\b",
            clean_log,
            re.IGNORECASE,
        )
    ]

    placement = [
        tuple(map(int, match))
        for match in re.findall(
            r"model weight placement:\s*requested_gpu_layers=(-?\d+)\s+"
            r"gpu_offload_supported=(\d+)\s+total_tensors=(\d+)\s+"
            r"cuda_tensors=(\d+)\s+cuda_tensor_bytes=(\d+)",
            clean_log,
            re.IGNORECASE,
        )
    ]
    has_cuda_weight_tensors = any(
        requested_layers != 0
        and gpu_supported == 1
        and total_tensors > 0
        and cuda_tensors > 0
        and cuda_bytes > 0
        for requested_layers, gpu_supported, total_tensors, cuda_tensors, cuda_bytes in placement
    )
    has_cuda_buffer = any(size > 0.0 for size in cuda_buffer_sizes)
    has_cuda_device_and_offload = (
        any(count > 0 for count in device_counts)
        and any(offloaded > 0 and total > 0 for offloaded, total in offloaded_layers)
    )
    if has_cuda_weight_tensors or has_cuda_buffer or has_cuda_device_and_offload:
        return

    diagnostics = [
        line.strip()
        for line in clean_log.splitlines()
        if re.search(
            r"CUDA|model buffer size|offloaded .* layers to GPU|model weight placement",
            line,
            re.IGNORECASE,
        )
    ]
    if not diagnostics:
        diagnostics = ["<no CUDA device, buffer, or layer-offload diagnostics were emitted>"]
    details = "\n  ".join(diagnostics[-12:])
    raise RuntimeError(
        f"{description} did not confirm CUDA model offload; "
        f"CUDA buffer sizes={cuda_buffer_sizes}, devices={device_counts}, "
        f"offloaded layers={offloaded_layers}. Relevant log lines:\n  {details}"
    )


def require_backward_precision(log, precision, description):
    if not re.search(rf"requested backward precision={re.escape(precision)}\b", log):
        raise RuntimeError(f"{description} did not report backward precision {precision}")


def extract_last_perplexity(log):
    values = re.findall(r"\[(\d+)\]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", log)
    if not values:
        raise RuntimeError("perplexity output did not contain any strided perplexity values")
    value = float(values[-1][1])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"perplexity output is not finite and positive: {value}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", required=True, type=Path)
    parser.add_argument("--perplexity", type=Path)
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--backward-precision", choices=("f32", "f16", "bf16"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--verbose-subprocess-logs", action="store_true",
        default=os.environ.get("QLORA_TEST_VERBOSE_LOGS") == "1",
        help="print every line from model, trainer, and perplexity subprocesses",
    )
    args = parser.parse_args()
    if args.require_cuda and args.n_gpu_layers <= 0:
        raise RuntimeError("--require-cuda requires --n-gpu-layers greater than zero")

    executables = [(args.trainer, "QLoRA trainer")]
    if not args.skip_perplexity:
        if args.perplexity is None:
            raise RuntimeError("--perplexity is required unless --skip-perplexity is set")
        executables.append((args.perplexity, "perplexity tool"))
    for executable, label in executables:
        if not executable.is_file():
            raise RuntimeError(f"{label} executable does not exist: {executable}")
    if not args.model.is_file():
        raise RuntimeError(f"GGUF model does not exist: {args.model}")
    with args.model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError(f"model does not have GGUF magic: {args.model}")

    backend_label = "CUDA" if args.require_cuda else "CPU"
    temp_prefix = "qlora-cuda-integration-" if args.require_cuda else "qlora-cpu-integration-"
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as tmp:
        work = Path(tmp)
        train_file = work / "train.jsonl"
        eval_file = work / "eval.txt"
        output_dir = work / "output"
        adapter_file = output_dir / "adapter.gguf"
        train_file.write_text(
            "".join(json.dumps({"text": text}) + "\n" for text in TRAIN_TEXTS),
            encoding="utf-8",
        )
        if not args.skip_perplexity:
            eval_file.write_text(EVAL_TEXT + "\n", encoding="utf-8")

        common = [
            "--model", args.model,
            "-t", "1",
            "-ngl", str(args.n_gpu_layers),
            "-opt", "adamw",
            "--optimizer-state-type", "f32",
            "-c", "300",
            "-b", "300",
            "-ub", "300",
        ]
        if not args.skip_perplexity:
            eval_common = [
                "--model", args.model,
                "--file", eval_file,
                "-t", "1",
                "-ngl", str(args.n_gpu_layers),
                "-c", "32",
                "-b", "32",
                "--ppl-stride", "16",
            ]
            baseline_log = run(
                [args.perplexity, *eval_common], args.timeout, f"baseline {backend_label} perplexity",
                verbose=args.verbose_subprocess_logs,
            )
            extract_last_perplexity(baseline_log)
            if args.require_cuda:
                require_cuda_offload(baseline_log, "baseline perplexity")

        precision_args = ["--backward-precision", args.backward_precision] if args.backward_precision else []
        train_log = run(
            [
                args.trainer,
                *common,
                *precision_args,
                "--train-file", train_file,
                "--output-dir", output_dir,
                "--lora-rank", "2",
                "--lora-alpha", "2",
                "--learning-rate", "1e-5",
                "--epochs", "1",
                "--max-steps", "2",
                "--val-windows", "0",
                "--gradient-accumulation-steps", "2",
                "--save-snapshot-steps", "1",
                "--save-snapshot-limit", "1",
                "--save-checkpoint-steps", "1",
                "--save-checkpoint-limit", "2",
                "--shuffle-dataset",
                "--seed", "42",
            ],
            args.timeout,
            f"one-epoch {backend_label} QLoRA training",
            verbose=args.verbose_subprocess_logs,
        )
        if args.require_cuda:
            require_cuda_offload(train_log, "uninterrupted training")
            require_backward_precision(train_log, "f16", "uninterrupted training")
        require_gguf(adapter_file, "trained LoRA adapter")

        prepared = re.search(r"prepared\s+\d+\s+tokens into\s+(\d+)\s+windows", train_log)
        if not prepared or int(prepared.group(1)) < 4:
            raise RuntimeError("resume-equivalence fixture did not produce at least four training windows")
        context = re.search(r"n_ctx=(\d+) n_ubatch=(\d+) stride=(\d+)", train_log)
        if not context or tuple(map(int, context.groups())) != (300, 300, 150):
            raise RuntimeError("non-256-aligned context was not preserved by full-context training")
        accumulation = re.search(
            r"dataset: (\d+) full-context windows; accumulate 2 windows/update = (\d+) optimizer steps/epoch",
            train_log,
        )
        if not accumulation:
            raise RuntimeError("training log did not report the requested gradient accumulation")
        windows, optimizer_steps_per_epoch = map(int, accumulation.groups())
        if optimizer_steps_per_epoch < 2:
            raise RuntimeError("resume-equivalence fixture must contain at least two optimizer steps per epoch")
        loss_match = re.search(r"epoch\s+1/1:\s+train_loss=([^\s]+)", train_log)
        if not loss_match:
            raise RuntimeError("training log did not report the epoch loss")
        train_loss = float(loss_match.group(1))
        if not math.isfinite(train_loss) or train_loss < 0:
            raise RuntimeError(f"training loss is not finite and non-negative: {train_loss}")

        if not re.search(r"saved checkpoint and snapshot \(optimizer step 1\)", train_log) or not re.search(
            r"saved checkpoint and snapshot \(optimizer step 2\)", train_log
        ):
            raise RuntimeError("training did not save both checkpoints required for resume equivalence")
        checkpoint_logs = re.findall(
            r"saved checkpoint(?: and snapshot)? \(optimizer step (\d+)\) "
            r"avg_loss=([^\s]+) elapsed=(\d{2,}:\d{2}:\d{2})",
            train_log,
        )
        if not checkpoint_logs:
            raise RuntimeError("checkpoint save logs did not report average loss and elapsed time")
        for step, average_loss, elapsed in checkpoint_logs:
            if not math.isfinite(float(average_loss)) or float(average_loss) < 0:
                raise RuntimeError(f"checkpoint {step} reported an invalid average loss: {average_loss}")
            if not re.fullmatch(r"\d{2,}:\d{2}:\d{2}", elapsed):
                raise RuntimeError(f"checkpoint {step} reported an invalid elapsed time: {elapsed}")
        saved_snapshots = sorted(output_dir.glob("snapshot-*.gguf"))
        if not saved_snapshots:
            raise RuntimeError("training did not save a periodic adapter snapshot")
        snapshot = saved_snapshots[-1]
        require_gguf(snapshot, "periodic LoRA snapshot")
        checkpoint_one = output_dir / "checkpoint-000001"
        checkpoint_two = output_dir / "checkpoint-000002"
        checkpoint_adapter = checkpoint_two / "adapter.gguf"
        require_gguf(checkpoint_one / "adapter.gguf", "intermediate checkpoint adapter")
        require_gguf(checkpoint_adapter, "final checkpoint adapter")
        checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
        if len(checkpoints) != 2 or len(saved_snapshots) != 1:
            raise RuntimeError("checkpoint or snapshot retention limit was not enforced")

        checkpoint_one_state = json.loads((checkpoint_one / "training-state.json").read_text(encoding="utf-8"))
        if checkpoint_one_state.get("format_version") != 4:
            raise RuntimeError("checkpoint did not use the epoch-statistics format")
        if checkpoint_one_state.get("training_statistics_complete") is not True:
            raise RuntimeError("checkpoint did not mark its epoch statistics as complete")
        if checkpoint_one_state.get("training_loss_sum", 0.0) <= 0.0:
            raise RuntimeError("checkpoint did not persist the accumulated training loss")
        if checkpoint_one_state.get("next_window") != 2:
            raise RuntimeError("first checkpoint was not saved in the middle of an epoch")
        if checkpoint_one_state.get("training_data_count") != 2 * 300:
            raise RuntimeError("checkpoint did not persist the processed training data count")
        if checkpoint_one_state.get("training_weight_sum", 0.0) <= 0.0:
            raise RuntimeError("checkpoint did not persist the accumulated supervised weight")
        if adapter_file.read_bytes() != checkpoint_adapter.read_bytes():
            raise RuntimeError("final adapter does not match the final checkpoint adapter")
        if snapshot.read_bytes() != checkpoint_adapter.read_bytes():
            raise RuntimeError("same-step snapshot does not match the checkpoint adapter")
        link_probe = output_dir / ".hardlink-probe"
        try:
            os.link(checkpoint_adapter, link_probe)
            hardlinks_supported = os.path.samefile(checkpoint_adapter, link_probe)
        except OSError:
            hardlinks_supported = False
        finally:
            link_probe.unlink(missing_ok=True)
        if hardlinks_supported and not all(
            os.path.samefile(path, checkpoint_adapter)
            for path in (adapter_file, snapshot)
        ):
            raise RuntimeError("adapter artifacts were copied despite hard-link support")

        def progress_rows(log):
            pattern = (
                r"\[0*(\d+)/0*\d+ \([^\]]+\)\] "
                r"loss=([^\s]+) avg=([^\s]+) supervised=(\d+) weight_sum=([^\s]+)"
            )
            matches = re.findall(pattern, log)
            if not matches:
                raise RuntimeError("training log did not report progress loss statistics")

            rows = [
                {
                    "window": int(window),
                    "loss": float(loss),
                    "average": float(average),
                    "supervised": int(supervised),
                    "weight_sum": float(weight_sum),
                }
                for window, loss, average, supervised, weight_sum in matches
            ]
            weighted_loss_sum = 0.0
            total_weight_sum = 0.0
            for row in rows:
                weighted_loss_sum += row["loss"] * row["weight_sum"]
                total_weight_sum += row["weight_sum"]
                expected_average = weighted_loss_sum / total_weight_sum if total_weight_sum > 0 else 0.0
                if not math.isclose(
                    row["average"], expected_average, rel_tol=2e-3, abs_tol=2e-3,
                ):
                    raise RuntimeError(
                        f"progress avg at window {row['window']} is not the supervision-weighted mean: "
                        f"reported={row['average']}, expected={expected_average}"
                    )
            return rows

        def progress_metrics(log, window):
            pattern = rf"\[0*{window}/0*\d+ \([^\]]+\)\] loss=([^\s]+) avg=([^\s]+)"
            matches = re.findall(pattern, log)
            if not matches:
                raise RuntimeError(f"training log did not report loss and average at window {window}")
            return tuple(float(value) for value in matches[0])

        uninterrupted_progress = progress_rows(train_log)
        checkpoint_average_by_step = {}
        for step, average_loss, _ in checkpoint_logs:
            checkpoint_average_by_step.setdefault(int(step), float(average_loss))
        for step, window in ((1, 2), (2, 4)):
            progress_average = next(
                (row["average"] for row in uninterrupted_progress if row["window"] == window), None
            )
            checkpoint_average = checkpoint_average_by_step.get(step)
            if progress_average is None or checkpoint_average is None or not math.isclose(
                checkpoint_average, progress_average, rel_tol=2e-3, abs_tol=2e-3,
            ):
                raise RuntimeError(
                    f"checkpoint {step} average loss does not match progress at window {window}"
                )
        if not math.isclose(
            train_loss, uninterrupted_progress[-1]["average"], rel_tol=2e-3, abs_tol=2e-3,
        ):
            raise RuntimeError(
                "epoch train loss differs from the final supervision-weighted progress average: "
                f"epoch={train_loss}, progress={uninterrupted_progress[-1]['average']}"
            )
        uninterrupted_loss, uninterrupted_average = progress_metrics(train_log, 3)

        resumed_output_dir = work / "resumed-output"
        resumed_log = run(
            [
                args.trainer,
                *common,
                *precision_args,
                "--train-file", train_file,
                "--output-dir", resumed_output_dir,
                "--lora-rank", "2",
                "--lora-alpha", "2",
                "--learning-rate", "1e-5",
                "--epochs", "1",
                "--max-steps", "2",
                "--gradient-accumulation-steps", "2",
                "--save-snapshot-steps", "1",
                "--save-snapshot-limit", "1",
                "--save-checkpoint-steps", "1",
                "--save-checkpoint-limit", "2",
                "--val-windows", "1",
                "--seed", "42",
                "--resume-from", checkpoint_one,
            ],
            args.timeout,
            f"resumed {backend_label} QLoRA training",
            verbose=args.verbose_subprocess_logs,
        )
        if args.require_cuda:
            require_cuda_offload(resumed_log, "resumed training")
            require_backward_precision(resumed_log, "f16", "resumed training")
        resumed_loss, resumed_average = progress_metrics(resumed_log, 3)
        if not math.isclose(uninterrupted_loss, resumed_loss, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "loss at the post-checkpoint optimizer update differs: "
                f"continuous={uninterrupted_loss}, resumed={resumed_loss}"
            )
        if not math.isclose(uninterrupted_average, resumed_average, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "resumed progress average did not include the pre-checkpoint epoch statistics: "
                f"continuous={uninterrupted_average}, resumed={resumed_average}"
            )

        uninterrupted_epoch_loss = re.search(r"epoch\s+1/1:\s+train_loss=([^\s]+)", train_log)
        resumed_epoch_loss = re.search(r"epoch\s+1/1:\s+train_loss=([^\s]+)", resumed_log)
        if not uninterrupted_epoch_loss or not resumed_epoch_loss:
            raise RuntimeError("continuous or resumed training did not report the epoch loss")
        if not math.isclose(
            float(uninterrupted_epoch_loss.group(1)), float(resumed_epoch_loss.group(1)),
            rel_tol=0.0, abs_tol=1e-6,
        ):
            raise RuntimeError("resumed epoch loss differs from uninterrupted training")

        resumed_adapter = resumed_output_dir / "adapter.gguf"
        resumed_checkpoint = resumed_output_dir / "checkpoint-000002"
        require_gguf(resumed_adapter, "resumed LoRA adapter")
        require_gguf(resumed_checkpoint / "adapter.gguf", "resumed final checkpoint adapter")
        if resumed_adapter.read_bytes() != checkpoint_adapter.read_bytes():
            adapter_diff = compare_gguf_tensor_values(checkpoint_adapter, resumed_adapter)
            optimizer_diff = compare_optimizer_state_values(
                checkpoint_two / "optimizer.gguf", resumed_checkpoint / "optimizer.gguf",
            )
            raise RuntimeError(
                "resumed adapter differs from uninterrupted training; "
                f"adapter tensors: {format_tensor_differences(adapter_diff)}; "
                f"optimizer state: {format_tensor_differences(optimizer_diff)}"
            )
        if (resumed_checkpoint / "optimizer.gguf").read_bytes() != (checkpoint_two / "optimizer.gguf").read_bytes():
            optimizer_diff = compare_optimizer_state_values(
                checkpoint_two / "optimizer.gguf", resumed_checkpoint / "optimizer.gguf",
            )
            raise RuntimeError(
                "resumed optimizer state differs from uninterrupted training; "
                f"optimizer tensors: {format_tensor_differences(optimizer_diff)}"
            )

        resumed_state = json.loads((resumed_checkpoint / "training-state.json").read_text(encoding="utf-8"))
        uninterrupted_state = json.loads((checkpoint_two / "training-state.json").read_text(encoding="utf-8"))
        for key in ("next_epoch", "next_window", "scheduler_step", "dataset_permutation",
                    "training_data_count", "training_weight_sum", "training_loss_sum",
                    "training_statistics_complete"):
            if resumed_state.get(key) != uninterrupted_state.get(key):
                raise RuntimeError(f"resumed checkpoint metadata differs for {key}")

        changed_output_dir = work / "changed-resume-output"
        changed_log = run(
            [
                args.trainer,
                *common,
                "--train-file", train_file,
                "--output-dir", changed_output_dir,
                "--lora-rank", "2",
                "--lora-alpha", "2",
                "--lora-type", "f16",
                "--learning-rate", "2e-5",
                "--learning-rate-min", "1e-6",
                "--weight-decay", "0.01",
                "--epochs", "2",
                "--max-steps", "3",
                "--val-windows", "1",
                "--gradient-accumulation-steps", "2",
                "--lr-scheduler", "cosine",
                "--warmup-steps", "1",
                "--warmup-init-ratio", "0.2",
                "--forward-precision", "f16",
                "--backward-precision", "f16",
                "--save-checkpoint-limit", "1",
                "--resume-from", checkpoint_one,
            ],
            args.timeout,
            f"resumed {backend_label} QLoRA training with changed compatible settings",
            verbose=args.verbose_subprocess_logs,
        )
        if args.require_cuda:
            require_cuda_offload(changed_log, "resumed training with changed settings")
            require_backward_precision(changed_log, "f16", "resumed training with changed settings")
        changed_checkpoint = changed_output_dir / "checkpoint-000003"
        require_gguf(changed_output_dir / "adapter.gguf", "adapter from resumed training with changed settings")
        require_gguf(changed_checkpoint / "adapter.gguf", "checkpoint adapter from resumed training with changed settings")
        changed_state = json.loads((changed_checkpoint / "training-state.json").read_text(encoding="utf-8"))
        expected_changed_settings = {
            "lora_type": "f16",
            "lr_scheduler": "cosine",
            "warmup_steps": 1,
            "max_steps": 3,
            "val_windows": 1,
            "forward_precision": "f16",
            "backward_precision": "f16",
        }
        for key, expected in expected_changed_settings.items():
            if changed_state.get(key) != expected:
                raise RuntimeError(f"resumed checkpoint did not persist changed setting {key}")
        for key, expected in {
            "learning_rate": 2e-5,
            "learning_rate_min": 1e-6,
            "weight_decay": 0.01,
            "warmup_init_ratio": 0.2,
        }.items():
            if not math.isclose(float(changed_state.get(key, math.nan)), expected, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(f"resumed checkpoint did not persist changed setting {key}")
        if changed_state.get("scheduler_step") != 3:
            raise RuntimeError("resume with changed training settings did not reach the new max-steps limit")

        if not args.skip_perplexity:
            adapter_log = run(
                [args.perplexity, *eval_common, "--lora", adapter_file],
                args.timeout,
                f"trained-adapter {backend_label} perplexity",
                verbose=args.verbose_subprocess_logs,
            )
            extract_last_perplexity(adapter_log)
            if args.require_cuda:
                require_cuda_offload(adapter_log, "trained adapter perplexity")

        print(f"[PASS] {backend_label} model-backed training and resume tests", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        sys.exit(1)
