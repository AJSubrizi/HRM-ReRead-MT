"""Augment a JSONL dataset with teacher completions.

Input rows should contain at least a ``prompt`` field. The output preserves all
fields and adds ``teacher_response``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from .teacher_utils import DeepSeekTeacherClient


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def augment_file(input_path: Path, output_path: Path, *, limit: Optional[int] = None) -> int:
    client = DeepSeekTeacherClient()
    rows = iter_jsonl(input_path)
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="augmenting"):
            if limit is not None and count >= limit:
                break
            prompt = row.get("prompt") or row.get("instruction") or row.get("question")
            if not prompt:
                raise ValueError("Each row must include prompt, instruction, or question.")
            teacher = client.complete(str(prompt))
            row["teacher_response"] = teacher.text
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = augment_file(args.input, args.output, limit=args.limit)
    print(f"Wrote {count} augmented rows to {args.output}")


if __name__ == "__main__":
    main()
