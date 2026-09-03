"""Create deterministic, prompt-family-aware train, validation, and test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from validate_dataset import review_reasons  # noqa: E402

from fit_chatbot.data import (  # noqa: E402
    classify_risk,
    classify_topic,
    load_json_list,
    parse_record,
    write_json,
)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def family_key(instruction: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", instruction.casefold())
    content = sorted({token for token in tokens if token not in STOP_WORDS})
    return f"{classify_topic(instruction)}:{' '.join(content)}"


def split_name(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "fitness_qa.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset" / "splits")
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="Include medically sensitive or heuristically questionable records in model training.",
    )
    args = parser.parse_args()

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": [], "review": []}
    family_assignments: dict[str, str] = {}
    for index, item in enumerate(load_json_list(args.dataset)):
        record, _ = parse_record(item, strip_boilerplate=False)
        if record is None:
            continue
        family = family_key(record.instruction)
        assigned = family_assignments.setdefault(family, split_name(family))
        reasons = review_reasons(record.instruction, record.output)
        row = {
            "id": hashlib.sha256(record.instruction.encode("utf-8")).hexdigest()[:16],
            "messages": record.to_messages(),
            "metadata": {
                "source_index": index,
                "topic": classify_topic(record.instruction),
                "risk": classify_risk(record.instruction),
                "family": family,
                "review_status": "needs_sme_review" if reasons else "auto_screened",
                "review_reasons": reasons,
                "schema_version": 2,
            },
        }
        if reasons and not args.include_needs_review:
            splits["review"].append(row)
        else:
            splits[assigned].append(row)

    for name, rows in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    manifest = {
        "strategy": "stable SHA-256 assignment by normalized prompt family",
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "topics": {
            name: dict(sorted(Counter(row["metadata"]["topic"] for row in rows).items()))
            for name, rows in splits.items()
        },
        "include_needs_review": args.include_needs_review,
        "family_leakage": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
