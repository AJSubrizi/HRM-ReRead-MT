"""Generate a tiny synthetic arithmetic dataset for local smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_dataset(output: Path, rows: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            left = index % 20
            right = (index * 7) % 20
            answer = left + right
            record = {
                "prompt": f"Solve {left} + {right}.",
                "teacher_response": (
                    f"Add {left} and {right}: {left} + {right} = {answer}. The answer is {answer}."
                ),
            }
            handle.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_math_100.jsonl"))
    parser.add_argument("--rows", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_dataset(args.output, args.rows)
    print(f"Wrote {args.rows} rows to {args.output}")


if __name__ == "__main__":
    main()
