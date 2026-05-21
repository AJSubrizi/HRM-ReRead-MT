#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python3}" -m hrm_reread_mt.train_re_read_mt \
  --data examples/sample_augmented.jsonl \
  --epochs 5 \
  --hidden-size 32 \
  --max-length 96 \
  --seed 7
