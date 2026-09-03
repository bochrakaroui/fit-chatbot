"""Extract manual additions and edits from the legacy combined dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fit_chatbot.data import (  # noqa: E402
    build_records,
    instruction_key,
    load_json_list,
    parse_record,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, default=REPO_ROOT / "fitness_qa.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dataset" / "curated" / "overrides.json",
    )
    parser.add_argument("--git-ref", help="Read the combined dataset from this Git revision")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=REPO_ROOT / "dataset" / "curated" / "exclusions.json",
    )
    args = parser.parse_args()

    raw, _ = build_records(REPO_ROOT / "raw_batches", strip_boilerplate=False)
    raw_by_key = {instruction_key(record.instruction): record for record in raw}
    curated: dict[str, dict[str, str]] = {}
    seen_current: set[str] = set()

    if args.git_ref:
        content = subprocess.check_output(
            ["git", "show", f"{args.git_ref}:fitness_qa.json"],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
        )
        combined_rows = json.loads(content)
    else:
        combined_rows = load_json_list(args.combined)

    for item in combined_rows:
        record, _ = parse_record(item, strip_boilerplate=False)
        if record is None:
            continue
        key = instruction_key(record.instruction)
        if key in seen_current:
            continue
        seen_current.add(key)
        if key not in raw_by_key or record != raw_by_key[key]:
            curated[key] = record.to_dict()

    write_json(args.output, list(curated.values()))
    exclusions = [record.instruction for key, record in raw_by_key.items() if key not in seen_current]
    write_json(args.exclusions, exclusions)
    additions = sum(key not in raw_by_key for key in curated)
    print(
        f"Wrote {len(curated)} curated records ({additions} additions) and "
        f"{len(exclusions)} exclusions"
    )


if __name__ == "__main__":
    main()
