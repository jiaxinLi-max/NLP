#!/usr/bin/env bash
# End-to-end smoke test: load model, run a few GSM8K examples with and without
# token pruning, print baseline vs pruned EM and KV cache MB.
#
# Requirements: free T4 / 16GB GPU; logged in to HuggingFace (gated Llama2 weights).
#   huggingface-cli login

set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "=== Smoke test: 5 GSM8K examples, baseline vs token_recent_distant @ 50% ==="
python scripts/run_gsm8k.py \
    --strategy none \
    --num-examples 5 \
    --output results/smoke_baseline.json

python scripts/run_gsm8k.py \
    --strategy token_recent_distant \
    --keep-ratio 0.5 \
    --num-examples 5 \
    --output results/smoke_token_prune.json

echo ""
echo "=== Smoke test complete. See results/smoke_*.json ==="
