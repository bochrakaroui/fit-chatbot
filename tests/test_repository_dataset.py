from __future__ import annotations

import json
from pathlib import Path

from fit_chatbot.data import SAFETY_SUFFIX, build_records, instruction_key

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_dataset_is_reproducible() -> None:
    records, _ = build_records(
        ROOT / "raw_batches",
        ROOT / "dataset" / "curated" / "overrides.json",
        ROOT / "dataset" / "curated" / "exclusions.json",
    )
    committed = json.loads((ROOT / "fitness_qa.json").read_text(encoding="utf-8"))
    assert [record.to_dict() for record in records] == committed


def test_canonical_dataset_has_no_exact_duplicates_or_boilerplate() -> None:
    payload = json.loads((ROOT / "fitness_qa.json").read_text(encoding="utf-8"))
    keys = [instruction_key(item["instruction"]) for item in payload]
    assert len(keys) == len(set(keys))
    assert all(SAFETY_SUFFIX not in item["output"] for item in payload)


def test_prompt_families_do_not_cross_model_splits() -> None:
    family_sets = {}
    for name in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (ROOT / "dataset" / "splits" / f"{name}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        family_sets[name] = {row["metadata"]["family"] for row in rows}
    assert family_sets["train"].isdisjoint(family_sets["validation"])
    assert family_sets["train"].isdisjoint(family_sets["test"])
    assert family_sets["validation"].isdisjoint(family_sets["test"])
