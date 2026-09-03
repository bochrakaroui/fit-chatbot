"""Validate the canonical dataset and produce an actionable review queue."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fit_chatbot.data import (  # noqa: E402
    SAFETY_SUFFIX,
    classify_risk,
    classify_topic,
    instruction_key,
    load_json_list,
    parse_record,
    semantic_key,
    write_json,
)

QUESTIONABLE_PATTERNS = {
    "knee_over_toe_absolute": re.compile(r"knees?.{0,35}(?:behind|not.*past|extend past).{0,20}toes?", re.I),
    "fixed_ninety_degree_limit": re.compile(r"(?:not recommended|avoid).{0,35}90[- ]degree", re.I),
    "heels_off_ground": re.compile(r"heels?.{0,25}(?:off|lift).{0,15}(?:ground|floor)", re.I),
    "spot_reduction": re.compile(r"(?:lose|burn|reduce).{0,20}(?:belly|stomach|arm|thigh) fat", re.I),
}


def review_reasons(instruction: str, output: str) -> list[str]:
    reasons: list[str] = []
    risk = classify_risk(instruction)
    if risk != "low":
        reasons.append(f"{risk}_risk_topic")
    for name, pattern in QUESTIONABLE_PATTERNS.items():
        if pattern.search(output):
            reasons.append(name)
    return reasons


def validate(dataset_path: Path, report_path: Path, review_path: Path) -> dict:
    payload = load_json_list(dataset_path)
    records = []
    invalid = []
    for index, item in enumerate(payload):
        record, _ = parse_record(item, strip_boilerplate=False)
        if record is None:
            invalid.append(index)
        else:
            records.append(record)

    exact = Counter(instruction_key(record.instruction) for record in records)
    semantic = Counter(semantic_key(record.instruction) for record in records)
    review_queue = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        reasons = review_reasons(record.instruction, record.output)
        if reasons:
            reason_counts.update(reasons)
            review_queue.append(
                {
                    **record.to_dict(),
                    "topic": classify_topic(record.instruction),
                    "risk": classify_risk(record.instruction),
                    "review_reasons": reasons,
                    "review_status": "needs_sme_review",
                }
            )

    report = {
        "records": len(records),
        "invalid_record_indexes": invalid,
        "exact_duplicate_groups": sum(count > 1 for count in exact.values()),
        "semantic_duplicate_groups": sum(count > 1 for count in semantic.values()),
        "boilerplate_suffixes": sum(SAFETY_SUFFIX in record.output for record in records),
        "contextual_records": sum(bool(record.input) for record in records),
        "review_queue_records": len(review_queue),
        "review_reasons": dict(sorted(reason_counts.items())),
    }
    write_json(report_path, report)
    write_json(review_path, review_queue)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "fitness_qa.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "dataset" / "reports" / "validation_report.json",
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=REPO_ROOT / "dataset" / "reports" / "review_queue.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate(args.dataset, args.report, args.review_queue)
    print(json.dumps(report, indent=2))
    hard_failures = (
        len(report["invalid_record_indexes"])
        + report["exact_duplicate_groups"]
        + report["boilerplate_suffixes"]
    )
    if args.strict and hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
