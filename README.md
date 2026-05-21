# HRM-ReRead-MT

![Status](https://img.shields.io/badge/status-research_scaffold-blue)
![Empirical Claims](https://img.shields.io/badge/empirical_claims-none_yet-orange)
![Tests](https://img.shields.io/badge/tests-pytest-green)

Reference scaffolding for **Hierarchical Re-Reading with Multi-Teacher Latent Consolidation**.

> Status: Research scaffold — executable reference implementation, no empirical claims yet.

This repository is a research proposal plus a small executable Python reference implementation. It does
not yet contain a full 1B-parameter HRM training stack or empirical benchmark results.

## What This Is

- A minimal executable reference implementation of the memory/consolidation mechanism.
- A JSONL teacher-augmentation pipeline for building student training data.
- A PyTorch smoke trainer that demonstrates repeated re-reading with latent memory reuse.
- A research note describing the intended full method and the open validation work.

## What This Is Not

- Not a benchmarked HRM model.
- Not evidence of state-of-the-art performance.
- Not a full reproduction of a 1B-parameter training stack.
- Not a claim that the projected gains in `PAPER.md` have been measured.

## What Is Here

- `PAPER.md`: research note and mathematical formulation.
- `src/hrm_reread_mt/teacher_utils.py`: OpenAI-compatible DeepSeek teacher client.
- `src/hrm_reread_mt/data_augment.py`: JSONL augmentation CLI.
- `src/hrm_reread_mt/train_re_read_mt.py`: laptop-sized reference loop showing latent memory across re-read epochs.
- `examples/`: sample raw/augmented JSONL plus a no-API smoke script.
- `configs/`: lightweight experiment config sketches.
- `tests/`: smoke tests for the reference loop.

## Architecture

```text
Input JSONL
   |
   v
Teacher Augmentation
   |
   v
Student Re-Read Epoch 1
   |
   v latent memory
Student Re-Read Epoch 2..N
   |
   v consolidation loss
Evaluation
```

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

Create a `.env` from `.env.example` and set `DEEPSEEK_API_KEY`. The model is configurable:

```bash
DEEPSEEK_MODEL=deepseek-chat
```

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
examples/run_smoke.sh
```

Or run the bundled augmented sample directly:

```bash
hrm-reread-train --data examples/sample_augmented.jsonl --epochs 5 --hidden-size 32
```

## Smoke Result

This is a mechanical sanity check, not a benchmark.

```text
Dataset: 100 synthetic arithmetic QA rows
Model: TinyHierarchicalReader, hidden_size=128
Epochs: 10
Seed: 7
Observation: training loss decreased from 3.2868 to 0.1697
```

Reproduce it locally:

```bash
python3 examples/make_synthetic_math.py --output data/synthetic_math_100.jsonl --rows 100
hrm-reread-train --data data/synthetic_math_100.jsonl --epochs 10 --hidden-size 128 --seed 7
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
