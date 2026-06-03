"""Robust resumable downloader for the remaining safetensors shard.

Why: ModelScope's Python SDK keeps stalling near the end of large downloads
(~10KB/s). Direct HTTP GET with Range headers and chunked retries is more
reliable from China.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request


def download(url: str, dest: str, chunk: int = 1 << 20, total_timeout: int = 7200):
    deadline = time.time() + total_timeout
    while time.time() < deadline:
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "ab") as f:
                content_range = resp.headers.get("Content-Range", "")
                total = None
                if content_range and "/" in content_range:
                    total = int(content_range.split("/")[-1])
                else:
                    cl = resp.headers.get("Content-Length")
                    if cl is not None:
                        total = existing + int(cl)
                last_print = time.time()
                start_pos = existing
                start_t = time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    existing += len(buf)
                    now = time.time()
                    if now - last_print > 5:
                        rate = (existing - start_pos) / max(now - start_t, 1e-6) / 1e6
                        pct = 100.0 * existing / total if total else 0
                        print(f"  {existing/1e9:.2f}/{(total or 0)/1e9:.2f} GB ({pct:.1f}%) @ {rate:.2f} MB/s", flush=True)
                        last_print = now
            if total is None or os.path.getsize(dest) >= total:
                size = os.path.getsize(dest)
                print(f"  done: {size/1e9:.2f} GB")
                return size
        except Exception as e:
            print(f"  retry after error: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
            continue
    raise TimeoutError(f"failed to finish {url} within {total_timeout}s")


def main():
    target_dir = "/work/models/llama2-7b-chat"
    os.makedirs(target_dir, exist_ok=True)

    # ModelScope direct file URLs (no SDK).
    base = "https://modelscope.cn/models/shakechen/Llama-2-7b-chat-hf/resolve/master"
    shards = [
        ("model-00001-of-00002.safetensors", 9976570520),
        ("model-00002-of-00002.safetensors", 3500296424),
    ]
    # Shard 1 lives in ._____temp from a partial run; promote it.
    temp_dir = os.path.join(target_dir, "._____temp")
    for name, _ in shards:
        temp_path = os.path.join(temp_dir, name)
        final_path = os.path.join(target_dir, name)
        if os.path.exists(temp_path) and not os.path.exists(final_path):
            os.rename(temp_path, final_path)
            print(f"promoted partial: {name} ({os.path.getsize(final_path)/1e9:.2f} GB)")
        elif os.path.exists(temp_path):
            # Final exists; if temp larger, swap.
            if os.path.getsize(temp_path) > os.path.getsize(final_path):
                os.rename(temp_path, final_path + ".new")
                os.replace(final_path + ".new", final_path)
                print(f"replaced with larger temp: {name}")

    for name, _ in shards:
        path = os.path.join(target_dir, name)
        url = f"{base}/{name}"
        existing = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"\n{name}: starting at {existing/1e9:.2f} GB")
        download(url, path)

    print("\nAll shards present:")
    for name, _ in shards:
        print(f"  {name}: {os.path.getsize(os.path.join(target_dir, name))/1e9:.2f} GB")


if __name__ == "__main__":
    main()
