"""Merge a trained PEFT adapter into its base model for standalone inference."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the ML dependencies: pip install -e '.[ml]'") from exc

    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, device_map="auto")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)


if __name__ == "__main__":
    main()
