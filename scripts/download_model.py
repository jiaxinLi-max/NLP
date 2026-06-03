"""One-shot helper: pull Llama-2-7b-chat-hf from ModelScope into ./models/llama2-7b-chat.

Why ModelScope: HF connections from China are flaky (SSL EOF mid-stream) and
hf-mirror redirects 308 back to HF. ModelScope hosts the same weights at
shakechen/Llama-2-7b-chat-hf and downloads reliably.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from modelscope import snapshot_download


def main():
    target = Path(os.environ.get("MODEL_DIR", "/work/models/llama2-7b-chat"))
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {target}...")
    path = snapshot_download(
        "shakechen/Llama-2-7b-chat-hf",
        cache_dir=str(target.parent),
        local_dir=str(target),
        allow_file_pattern=[
            "*.json", "*.model", "*.txt", "*.md",
            "*.safetensors",
        ],
    )
    print(f"Done: {path}")
    files = sorted(p.name for p in target.iterdir())
    print(f"Files ({len(files)}): {files[:20]}")


if __name__ == "__main__":
    main()
