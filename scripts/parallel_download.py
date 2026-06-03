"""Parallel resumable HTTP downloader using Range headers + threads.

Splits the remaining range into N chunks, fetches them in parallel into
.partN sidecar files, then concatenates onto the existing prefix.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import urllib.request


URL = "https://modelscope.cn/models/shakechen/Llama-2-7b-chat-hf/resolve/master/model-00001-of-00002.safetensors"
DEST = os.environ.get(
    "DEST",
    os.path.join("models", "llama2-7b-chat", "model-00001-of-00002.safetensors"),
)
TOTAL = 9976570520  # known total
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))


def fetch_range(idx: int, start: int, end: int, part_path: str, status: dict):
    have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    real_start = start + have
    if real_start > end:
        status[idx] = end - start + 1
        return
    while True:
        try:
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={real_start}-{end}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, "ab") as f:
                while True:
                    buf = resp.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    status[idx] = have
            if os.path.getsize(part_path) >= end - start + 1:
                return
            real_start = start + os.path.getsize(part_path)
        except Exception as e:
            print(f"[part {idx}] err: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
            real_start = start + (os.path.getsize(part_path) if os.path.exists(part_path) else 0)


def main():
    existing = os.path.getsize(DEST) if os.path.exists(DEST) else 0
    print(f"existing: {existing/1e9:.2f} GB / total {TOTAL/1e9:.2f} GB", flush=True)
    if existing >= TOTAL:
        print("already complete")
        return
    remaining_start = existing
    chunk = (TOTAL - remaining_start + N_WORKERS - 1) // N_WORKERS

    parts = []
    threads = []
    status = {i: 0 for i in range(N_WORKERS)}
    sizes = []
    for i in range(N_WORKERS):
        s = remaining_start + i * chunk
        e = min(remaining_start + (i + 1) * chunk - 1, TOTAL - 1)
        if s > e:
            break
        part_path = DEST + f".part{i}"
        parts.append(part_path)
        sizes.append(e - s + 1)
        t = threading.Thread(target=fetch_range, args=(i, s, e, part_path, status), daemon=True)
        t.start()
        threads.append(t)

    # Reporter
    start_t = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        done = sum(status.values())
        rate = done / max(time.time() - start_t, 1) / 1e6
        pct = 100.0 * done / max(sum(sizes), 1)
        print(f"  {done/1e9:.2f}/{sum(sizes)/1e9:.2f} GB ({pct:.1f}%) @ {rate:.2f} MB/s", flush=True)
    for t in threads:
        t.join()

    # Concatenate
    print("concatenating...", flush=True)
    with open(DEST, "ab") as out:
        for p in parts:
            with open(p, "rb") as inf:
                while True:
                    buf = inf.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
            os.remove(p)
    print(f"final size: {os.path.getsize(DEST)/1e9:.2f} GB")
    if os.path.getsize(DEST) != TOTAL:
        sys.exit(f"size mismatch: got {os.path.getsize(DEST)} expected {TOTAL}")


if __name__ == "__main__":
    main()
