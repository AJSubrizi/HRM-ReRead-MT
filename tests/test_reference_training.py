from pathlib import Path

import pytest

from hrm_reread_mt.data_augment import augment_file, iter_jsonl
from hrm_reread_mt.teacher_utils import DeepSeekTeacherClient, TeacherClientError
from hrm_reread_mt.train_re_read_mt import load_texts, train_reference


def test_reference_training_smoke(tmp_path: Path) -> None:
    data = tmp_path / "sample.jsonl"
    data.write_text(
        '{"prompt": "2+2?", "teacher_response": "The answer is 4."}\n'
        '{"prompt": "3+5?", "teacher_response": "The answer is 8."}\n',
        encoding="utf-8",
    )

    history = train_reference(data, epochs=2, hidden_size=16, max_length=32)

    assert len(history.losses) == 2
    assert all(loss > 0 for loss in history.losses)


def test_load_texts_prefers_teacher_then_answer_then_prompt(tmp_path: Path) -> None:
    data = tmp_path / "mixed.jsonl"
    data.write_text(
        '{"prompt": "prompt only"}\n'
        '{"prompt": "prompt", "answer": "answer text"}\n'
        '{"prompt": "prompt", "answer": "answer", "teacher_response": "teacher text"}\n',
        encoding="utf-8",
    )

    assert load_texts(data) == ["prompt only", "answer text", "teacher text"]


def test_latent_memory_is_reused_after_first_epoch(tmp_path: Path) -> None:
    data = tmp_path / "sample.jsonl"
    data.write_text(
        '{"teacher_response": "The answer is 4."}\n{"teacher_response": "The answer is 8."}\n',
        encoding="utf-8",
    )

    history = train_reference(data, epochs=3, hidden_size=16, max_length=32)

    assert history.latent_was_reused == [False, True, True]
    assert history.consolidation_losses[0] == 0
    assert history.consolidation_losses[1] > 0


def test_invalid_jsonl_raises_clear_error(tmp_path: Path) -> None:
    data = tmp_path / "broken.jsonl"
    data.write_text('{"prompt": "ok"}\n{not-json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_texts(data)

    with pytest.raises(ValueError, match="Invalid JSONL"):
        list(iter_jsonl(data))


def test_augmentation_fails_cleanly_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    data = tmp_path / "raw.jsonl"
    data.write_text('{"prompt": "2+2?"}\n', encoding="utf-8")

    with pytest.raises(TeacherClientError, match="DEEPSEEK_API_KEY is not set"):
        augment_file(data, tmp_path / "augmented.jsonl")


def test_teacher_model_can_be_configured_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    client = DeepSeekTeacherClient()

    assert client.model == "deepseek-chat"
