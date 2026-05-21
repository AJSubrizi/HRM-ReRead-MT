from pathlib import Path

from hrm_reread_mt.train_re_read_mt import train_reference


def test_reference_training_smoke(tmp_path: Path) -> None:
    data = tmp_path / "sample.jsonl"
    data.write_text(
        '{"prompt": "2+2?", "teacher_response": "The answer is 4."}\n'
        '{"prompt": "3+5?", "teacher_response": "The answer is 8."}\n',
        encoding="utf-8",
    )

    losses = train_reference(data, epochs=2, hidden_size=16, max_length=32)

    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)
