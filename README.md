# HRM-ReRead-MT

Reference scaffolding for **Hierarchical Re-Reading with Multi-Teacher Latent Consolidation**.

This repository is a research proposal plus a small executable Python reference implementation. It does
not yet contain a full 1B-parameter HRM training stack or empirical benchmark results.

## What Is Here

- `PAPER.md`: research note and mathematical formulation.
- `src/hrm_reread_mt/teacher_utils.py`: OpenAI-compatible DeepSeek teacher client.
- `src/hrm_reread_mt/data_augment.py`: JSONL augmentation CLI.
- `src/hrm_reread_mt/train_re_read_mt.py`: laptop-sized reference loop showing latent memory across re-read epochs.
- `tests/`: smoke tests for the reference loop.

## Install

```bash
git clone https://github.com/AJSubrizi/HRM-ReRead-MT.git
cd HRM-ReRead-MT
bash setup.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data Augmentation

Create a `.env` from `.env.example` and set `DEEPSEEK_API_KEY`.

Input JSONL rows should contain one of `prompt`, `instruction`, or `question`:

```json
{"prompt": "Solve 2 + 2 and explain the reasoning."}
```

Run:

```bash
hrm-reread-augment --input data/raw.jsonl --output data/augmented.jsonl --limit 10
```

The output preserves each row and adds `teacher_response`.

## Reference Training Smoke Run

The included training command is intentionally tiny. It verifies the mechanics of:

- repeated re-reading epochs,
- a latent memory buffer,
- prediction loss plus consolidation loss.

```bash
hrm-reread-train --data data/augmented.jsonl --epochs 5
```

For a no-API local smoke test:

```bash
mkdir -p data
printf '{"prompt":"2+2?","teacher_response":"The answer is 4."}\n' > data/sample.jsonl
hrm-reread-train --data data/sample.jsonl --epochs 2 --hidden-size 16
```

## Development

```bash
pytest
ruff check .
```

## Roadmap

- Add a real HRM/HRM-Text adapter once the target base implementation is selected.
- Replace the character-level demo model with tokenizer-backed datasets.
- Add teacher-logit distillation where provider responses expose logprobs or compatible logits.
- Add experiment configs, checkpointing, evaluation scripts, and benchmark reporting.

## Status

Empirical validation is pending. Claims in `PAPER.md` should be read as hypotheses/projections until
benchmarks are run and published.

## License

MIT.
