#!/usr/bin/env python3
"""Opt-in CPU integration test for training and using a real GGUF LoRA adapter."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TRAIN_TEXTS = (
    "Apples are often red or green. Oranges are orange and bananas are yellow. "
    "Fruit grows on trees and provides useful food for people. Apples can be sweet, "
    "while lemons taste sour. A small language model predicts the next token from "
    "earlier words in its context. The model should process the same input consistently "
    "when the dataset, seed, and training settings stay fixed.",
    "A second training example describes the weather. Warm sunlight dries wet leaves, "
    "clouds can bring rain, and a strong wind moves branches. Simple sentences provide "
    "a separate sample so the integration test exercises more than one training window.",
)

EVAL_TEXT = " ".join(
    [
        "Small language models predict the next token from a sequence of earlier tokens.",
        "Training changes a small set of adapter weights while the base model remains frozen.",
        "A fixed evaluation corpus makes it possible to compare the model before and after an update.",
        "The purpose of this test is to verify that CPU training completes and the saved adapter loads for inference.",
    ]
    * 8
)


def print_subprocess_output(output, description, *, verbose, failed):
    if verbose:
        if output:
            print(output, flush=True)
        return

    lines = output.splitlines()
    keep = set()
    patterns = (
        re.compile(r"^\[CUDA-Q4-MATMUL\]"),
        re.compile(r"^\[DIAG(?:-[A-Za-z0-9_-]+)?\]"),
        re.compile(r"^\[(?:PASS|FAIL)\]"),
        re.compile(r"prepared\s+\d+\s+tokens into\s+\d+\s+windows", re.IGNORECASE),
        re.compile(r"epoch\s+\d+/\d+:\s+train_loss=", re.IGNORECASE),
        re.compile(r"periodic adapter snapshot saved", re.IGNORECASE),
        re.compile(r"\b(?:warning|warn|error|failed|failure)\b", re.IGNORECASE),
    )
    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in patterns):
            keep.add(index)

    if "perplexity" in description.lower():
        metric_indices = [index for index, line in enumerate(lines) if re.match(r"^\s*\[\d+\]\s*[0-9]", line)]
        if metric_indices:
            keep.add(metric_indices[-1])

    has_failure_detail = any(
        re.search(r"^\[FAIL\]|\b(?:error|failed|failure)\b", lines[index], re.IGNORECASE)
        for index in keep
    )
    if failed and not has_failure_detail:
        keep.update(range(max(0, len(lines) - 12), len(lines)))

    for index in sorted(keep):
        print(lines[index], flush=True)

    suppressed = len(lines) - len(keep)
    if suppressed > 0:
        print(
            f"[LOG] suppressed {suppressed} subprocess output lines; "
            "set QLORA_TEST_VERBOSE_LOGS=1 or pass --verbose-subprocess-logs to show all",
            flush=True,
        )


def run(command, timeout, description, *, verbose=False):
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
        print_subprocess_output(output, description, verbose=verbose, failed=True)
        raise RuntimeError(f"{description} timed out after {timeout} seconds") from exc

    print_subprocess_output(
        result.stdout or "", description, verbose=verbose, failed=result.returncode != 0,
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
    parser.add_argument("--perplexity", required=True, type=Path)
    parser.add_argument("--gradient-checker", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--verbose-subprocess-logs", action="store_true",
        default=os.environ.get("QLORA_TEST_VERBOSE_LOGS") == "1",
        help="print every line from model, trainer, and perplexity subprocesses",
    )
    args = parser.parse_args()

    for executable, label in (
        (args.trainer, "QLoRA trainer"),
        (args.perplexity, "perplexity tool"),
        (args.gradient_checker, "model gradient checker"),
    ):
        if not executable.is_file():
            raise RuntimeError(f"{label} executable does not exist: {executable}")
    if not args.model.is_file():
        raise RuntimeError(f"GGUF model does not exist: {args.model}")
    with args.model.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError(f"model does not have GGUF magic: {args.model}")

    with tempfile.TemporaryDirectory(prefix="qlora-cpu-integration-") as tmp:
        work = Path(tmp)
        train_file = work / "train.jsonl"
        eval_file = work / "eval.txt"
        adapter_file = work / "adapter.gguf"
        train_file.write_text(
            "".join(json.dumps({"text": text}) + "\n" for text in TRAIN_TEXTS),
            encoding="utf-8",
        )
        eval_file.write_text(EVAL_TEXT + "\n", encoding="utf-8")

        common = [
            "--model", args.model,
            "-t", "1",
            "-ngl", "0",
            "-c", "32",
            "-b", "32",
            "-ub", "32",
        ]
        eval_common = [
            "--model", args.model,
            "--file", eval_file,
            "-t", "1",
            "-ngl", "0",
            "-c", "32",
            "-b", "32",
            "--ppl-stride", "16",
        ]

        baseline_log = run(
            [args.perplexity, *eval_common], args.timeout, "baseline CPU perplexity",
            verbose=args.verbose_subprocess_logs,
        )
        baseline_ppl = extract_last_perplexity(baseline_log)

        gradient_checker_command = [args.gradient_checker, "--model", args.model]
        if args.verbose_subprocess_logs:
            gradient_checker_command.append("--verbose-logs")
        run(
            gradient_checker_command,
            args.timeout,
            "model-backed LoRA gradient finite-difference check",
            verbose=args.verbose_subprocess_logs,
        )

        train_log = run(
            [
                args.trainer,
                *common,
                "--train-file", train_file,
                "--lora-out", adapter_file,
                "--lora-rank", "2",
                "--lora-alpha", "2",
                "--lora-targets", "attn_q,attn_output,ffn_gate,ffn_up,ffn_down",
                "--learning-rate", "1e-5",
                "--epochs", "1",
                "--save-every", "1",
                "--seed", "42",
            ],
            args.timeout,
            "one-epoch CPU QLoRA training",
            verbose=args.verbose_subprocess_logs,
        )
        require_gguf(adapter_file, "trained LoRA adapter")

        prepared = re.search(r"prepared\s+\d+\s+tokens into\s+(\d+)\s+windows", train_log)
        if not prepared or int(prepared.group(1)) < 2:
            raise RuntimeError("integration fixture did not produce at least two training windows")
        loss_match = re.search(r"epoch\s+1/1:\s+train_loss=([^\s]+)", train_log)
        if not loss_match:
            raise RuntimeError("training log did not report the epoch loss")
        train_loss = float(loss_match.group(1))
        if not math.isfinite(train_loss) or train_loss < 0:
            raise RuntimeError(f"training loss is not finite and non-negative: {train_loss}")

        snapshots = re.findall(r"periodic adapter snapshot saved:\s*(.+?)\s+\(window\s+\d+\)", train_log)
        if not snapshots:
            raise RuntimeError("training did not save a periodic adapter snapshot")
        snapshot = Path(snapshots[-1].strip())
        require_gguf(snapshot, "periodic LoRA snapshot")

        adapter_log = run(
            [args.perplexity, *eval_common, "--lora", adapter_file],
            args.timeout,
            "trained-adapter CPU perplexity",
            verbose=args.verbose_subprocess_logs,
        )
        adapter_ppl = extract_last_perplexity(adapter_log)

        print(
            "[PASS] CPU model-backed training integration: "
            f"windows={prepared.group(1)}, train_loss={train_loss:.6g}, "
            f"baseline_perplexity={baseline_ppl:.6g}, adapter_perplexity={adapter_ppl:.6g}, "
            f"perplexity_delta={adapter_ppl - baseline_ppl:+.6g}, "
            f"adapter_bytes={adapter_file.stat().st_size}, snapshot={snapshot.name}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        sys.exit(1)
