#!/usr/bin/env python3
"""Download a pinned GGUF fixture used by the optional QLoRA CPU integration test."""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "q8_0": {
        "url": (
            "https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/"
            "resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf?download=true"
        ),
        "sha256": "40f7094960b6ede829145d102ca79451b364b27d9d8694d4406e002024cff357",
    },
    "q8_0_bartowski": {
        "url": (
            "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/"
            "resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf?download=true"
        ),
        "sha256": "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83",
    },
    "f16": {
        "url": (
            "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/"
            "resolve/main/SmolLM2-135M-Instruct-f16.gguf?download=true"
        ),
        "sha256": "f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="destination GGUF file")
    parser.add_argument(
        "--variant", choices=sorted(MODELS), default="q8_0",
        help="model weight type (use f16 to investigate whether quantization affects the gradient check; default: q8_0)",
    )
    args = parser.parse_args()
    model = MODELS[args.variant]
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        actual = sha256_file(destination)
        if actual == model["sha256"]:
            print(f"Verified existing {args.variant} model: {destination}")
            return
        raise RuntimeError(
            f"existing file checksum mismatch for {args.variant}: {destination} (sha256={actual})"
        )

    temporary = destination.with_name(destination.name + ".download")
    request = urllib.request.Request(model["url"], headers={"User-Agent": "llama.cpp-qlora-test-model/1.0"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else 0
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                received += len(block)
                if total_bytes:
                    print(f"\rDownloaded {received / total_bytes:.1%}", end="", flush=True)
            output.flush()
            os.fsync(output.fileno())
        print()
        actual = digest.hexdigest()
        if actual != model["sha256"]:
            raise RuntimeError(f"download checksum mismatch: expected {model['sha256']}, got {actual}")
        with temporary.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise RuntimeError("downloaded file does not have GGUF magic")
        temporary.replace(destination)
        print(f"Downloaded and verified {args.variant} model: {destination}")
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
