"""Build the canonical dataset from immutable batches and curated overrides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fit_chatbot.data import build_records, dataset_profile, sha256_file, write_json  # noqa: E402


def build(output: Path, report_path: Path, *, keep_boilerplate: bool = False) -> dict:
    records, report = build_records(
        REPO_ROOT / "raw_batches",
        REPO_ROOT / "dataset" / "curated" / "overrides.json",
        exclusions_path=REPO_ROOT / "dataset" / "curated" / "exclusions.json",
        strip_boilerplate=not keep_boilerplate,
    )
    write_json(output, [record.to_dict() for record in records])
    report.output_sha256 = sha256_file(output)
    payload = {"build": report.__dict__, "profile": dataset_profile(records)}
    write_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "fitness_qa.json")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "dataset" / "reports" / "build_report.json",
    )
    parser.add_argument("--keep-boilerplate", action="store_true")
    args = parser.parse_args()
    payload = build(args.output, args.report, keep_boilerplate=args.keep_boilerplate)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
