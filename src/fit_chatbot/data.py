"""Dataset schema, normalization, profiling, and deterministic build helpers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
REQUIRED_FIELDS = ("instruction", "input", "output")
SAFETY_SUFFIX = "If pain occurs, stop and reduce intensity."
SPACE_RE = re.compile(r"\s+")
PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class DatasetRecord:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_messages(self) -> list[dict[str, str]]:
        user_content = self.instruction
        if self.input:
            user_content = f"{self.instruction}\n\nUser context:\n{self.input}"
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": self.output},
        ]


@dataclass
class BuildReport:
    schema_version: int
    raw_files: int
    raw_rows: int
    curated_rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows_removed: int
    excluded_raw_rows: int
    curated_replacements: int
    curated_additions: int
    boilerplate_suffixes_removed: int
    contextual_rows: int
    output_rows: int
    output_sha256: str = ""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return SPACE_RE.sub(" ", text).strip()


def instruction_key(value: str) -> str:
    return normalize_text(value).casefold()


def semantic_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", instruction_key(value)).strip()


def clean_output(value: Any, *, strip_boilerplate: bool = True) -> tuple[str, bool]:
    text = normalize_text(value)
    removed = False
    if strip_boilerplate and text.endswith(SAFETY_SUFFIX):
        text = text[: -len(SAFETY_SUFFIX)].rstrip()
        removed = True
    text = PUNCTUATION_RE.sub(r"\1", text)
    text = re.sub(r"(?:\.\s*){2,}$", ".", text).strip()
    return text, removed


def parse_record(item: Any, *, strip_boilerplate: bool = True) -> tuple[DatasetRecord | None, bool]:
    if not isinstance(item, dict) or any(field not in item for field in REQUIRED_FIELDS):
        return None, False
    instruction = normalize_text(item["instruction"])
    context = normalize_text(item["input"])
    output, removed = clean_output(item["output"], strip_boilerplate=strip_boilerplate)
    if not instruction or not output:
        return None, removed
    return DatasetRecord(instruction, context, output), removed


def load_json_list(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON list")
    return payload


def build_records(
    raw_dir: Path,
    curated_path: Path | None = None,
    exclusions_path: Path | None = None,
    *,
    strip_boilerplate: bool = True,
) -> tuple[list[DatasetRecord], BuildReport]:
    files = sorted(raw_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON batches found in {raw_dir}")

    records: dict[str, DatasetRecord] = {}
    report = BuildReport(
        schema_version=SCHEMA_VERSION,
        raw_files=len(files),
        raw_rows=0,
        curated_rows=0,
        valid_rows=0,
        invalid_rows=0,
        duplicate_rows_removed=0,
        excluded_raw_rows=0,
        curated_replacements=0,
        curated_additions=0,
        boilerplate_suffixes_removed=0,
        contextual_rows=0,
        output_rows=0,
    )

    for path in files:
        for item in load_json_list(path):
            report.raw_rows += 1
            record, removed = parse_record(item, strip_boilerplate=strip_boilerplate)
            report.boilerplate_suffixes_removed += int(removed)
            if record is None:
                report.invalid_rows += 1
                continue
            report.valid_rows += 1
            key = instruction_key(record.instruction)
            if key in records:
                report.duplicate_rows_removed += 1
                continue
            records[key] = record

    if exclusions_path and exclusions_path.exists():
        exclusions = load_json_list(exclusions_path)
        for instruction in exclusions:
            key = instruction_key(str(instruction))
            if records.pop(key, None) is not None:
                report.excluded_raw_rows += 1

    if curated_path and curated_path.exists():
        for item in load_json_list(curated_path):
            report.curated_rows += 1
            record, removed = parse_record(item, strip_boilerplate=strip_boilerplate)
            report.boilerplate_suffixes_removed += int(removed)
            if record is None:
                report.invalid_rows += 1
                continue
            key = instruction_key(record.instruction)
            if key in records:
                report.curated_replacements += 1
            else:
                report.curated_additions += 1
            records[key] = record

    result = list(records.values())
    report.contextual_rows = sum(bool(record.input) for record in result)
    report.output_rows = len(result)
    return result, report


def dataset_profile(records: Iterable[DatasetRecord]) -> dict[str, Any]:
    rows = list(records)
    semantic_counts = Counter(semantic_key(record.instruction) for record in rows)
    topic_counts = Counter(classify_topic(record.instruction) for record in rows)
    risk_counts = Counter(classify_risk(record.instruction) for record in rows)
    lengths = sorted(len(record.output.split()) for record in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(rows),
        "contextual_records": sum(bool(record.input) for record in rows),
        "semantic_duplicate_groups": sum(count > 1 for count in semantic_counts.values()),
        "answer_words": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "p95": lengths[min(int(len(lengths) * 0.95), len(lengths) - 1)] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "topics": dict(sorted(topic_counts.items())),
        "risk_levels": dict(sorted(risk_counts.items())),
    }


def classify_topic(text: str) -> str:
    lowered = text.casefold()
    rules = (
        ("nutrition", ("calorie", "protein", "carb", "diet", "meal", "nutrition", "fat loss")),
        ("supplements", ("supplement", "creatine", "caffeine", "vitamin", "protein powder")),
        ("recovery", ("recover", "sleep", "sore", "rest day", "mobility")),
        ("programming", ("program", "routine", "sets", "reps", "frequency", "workout plan")),
        ("cardio", ("run", "cardio", "cycling", "swim", "aerobic")),
        ("exercise_form", ("form", "squat", "deadlift", "bench", "lunge", "push-up")),
    )
    for topic, words in rules:
        if any(word in lowered for word in words):
            return topic
    return "general_fitness"


def classify_risk(text: str) -> str:
    lowered = text.casefold()
    high = (
        "pregnan",
        "injury",
        "injured",
        "chest pain",
        "faint",
        "eating disorder",
        "medication",
        "medical condition",
        "diabetes",
        "heart",
    )
    medium = ("pain", "supplement", "dose", "dosage", "back issue", "knee issue")
    if any(term in lowered for term in high):
        return "high"
    if any(term in lowered for term in medium):
        return "medium"
    return "low"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
