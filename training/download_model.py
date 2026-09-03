"""Download a pinned model snapshot without loading it into memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--revision", default=None, help="Model branch, tag, or commit hash")
    parser.add_argument("--output", type=Path, default=Path("artifacts/models"))
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the ML dependencies: pip install -e '.[ml]'") from exc

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_name = config["model_name"]
    target = args.output / model_name.replace("/", "--")
    path = snapshot_download(
        repo_id=model_name,
        revision=args.revision,
        local_dir=target,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(path)


if __name__ == "__main__":
    main()
