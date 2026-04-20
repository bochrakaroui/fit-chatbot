import json
import json
from pathlib import Path


def merge_batches() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    raw_folder = repo_root / "raw_batches"
    output_path = repo_root / "fitness_qa.json"

    if not raw_folder.exists():
        raise FileNotFoundError(f"Could not find folder: {raw_folder}")

    all_examples = []
    skipped = 0

    # rglob covers all JSON files even if raw_batches later has subfolders.
    json_files = sorted(raw_folder.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {raw_folder}")

    for filepath in json_files:
        with filepath.open("r", encoding="utf-8") as f:
            content = f.read().strip()

        try:
            batch = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"SKIPPED {filepath.name} - invalid JSON: {e}")
            skipped += 1
            continue

        if not isinstance(batch, list):
            print(f"SKIPPED {filepath.name} - expected a list at top-level")
            skipped += 1
            continue

        valid = []
        for i, item in enumerate(batch):
            if not isinstance(item, dict):
                print(f"  Row {i} in {filepath.name} is not an object - skipped")
                continue

            if not all(k in item for k in ["instruction", "input", "output"]):
                print(f"  Row {i} in {filepath.name} missing keys - skipped")
                continue

            instruction = str(item["instruction"]).strip()
            output = str(item["output"]).strip()
            if not instruction or not output:
                print(f"  Row {i} in {filepath.name} has empty instruction/output - skipped")
                continue

            item["instruction"] = instruction
            item["input"] = ""
            item["output"] = output
            valid.append(item)

        all_examples.extend(valid)
        print(f"{filepath.name} - {len(valid)} examples added")

    seen = set()
    deduped = []
    for item in all_examples:
        key = item["instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    print(f"\nTotal before dedup: {len(all_examples)}")
    print(f"Total after dedup:  {len(deduped)}")
    print(f"Batches skipped:    {skipped}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    merge_batches()