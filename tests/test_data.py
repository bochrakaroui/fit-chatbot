from __future__ import annotations

import json
from pathlib import Path

from fit_chatbot.data import SAFETY_SUFFIX, build_records, clean_output, instruction_key, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n  world  ") == "hello world"


def test_clean_output_removes_repetitive_suffix() -> None:
    output, removed = clean_output(f"Use a controlled range. {SAFETY_SUFFIX}")
    assert output == "Use a controlled range."
    assert removed is True


def test_curated_records_replace_and_extend_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "batch.json").write_text(
        json.dumps(
            [
                {"instruction": "Question one?", "input": "", "output": "Old"},
                {"instruction": "Question two?", "input": "", "output": "Keep"},
            ]
        ),
        encoding="utf-8",
    )
    curated = tmp_path / "curated.json"
    curated.write_text(
        json.dumps(
            [
                {"instruction": "Question one?", "input": "context", "output": "New"},
                {"instruction": "Question three?", "input": "", "output": "Add"},
            ]
        ),
        encoding="utf-8",
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps(["Question two?"]), encoding="utf-8")

    records, report = build_records(raw, curated, exclusions)
    by_key = {instruction_key(record.instruction): record for record in records}
    assert set(by_key) == {"question one?", "question three?"}
    assert by_key["question one?"].input == "context"
    assert report.curated_replacements == 1
    assert report.curated_additions == 1
    assert report.excluded_raw_rows == 1
