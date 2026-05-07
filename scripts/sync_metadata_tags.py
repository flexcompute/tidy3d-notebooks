#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS_DIR = REPO_ROOT / "checks"
OUTPUT_PATH = CHECKS_DIR / "metadata_tags.json"


def collect_tags():
    applications = set()
    features = set()

    for notebook_path in sorted(REPO_ROOT.glob("*.ipynb")):
        with notebook_path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)

        metadata = notebook.get("metadata", {})
        for field_name, target in (("applications", applications), ("features", features)):
            values = metadata.get(field_name, [])
            if not isinstance(values, list):
                raise TypeError(f"{notebook_path.name}: metadata.{field_name} must be a list")
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{notebook_path.name}: metadata.{field_name} entries must be strings"
                    )
                stripped = value.strip()
                if stripped:
                    target.add(stripped)

    return {
        "applications": sorted(applications),
        "features": sorted(features),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate the catalog of notebook metadata applications/features tags."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if metadata_tags.json is not up to date instead of writing it.",
    )
    args = parser.parse_args()

    generated = collect_tags()
    rendered = json.dumps(generated, indent=2, ensure_ascii=True) + "\n"

    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print(
                "checks/metadata_tags.json is out of date. Run "
                "`uv run --no-project python scripts/sync_metadata_tags.py`.",
                file=sys.stderr,
            )
            return 1
        return 0

    CHECKS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
