#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

PATTERNS = [
    re.compile(r"/home/[^/\s\"']+"),
    re.compile(r"/Users/[^/\s\"']+"),
    re.compile(r"[A-Za-z]:(?:\\|\\\\)Users(?:\\|\\\\)[^\\/\s\"']+"),
]
REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORELIST_PATH = REPO_ROOT / "checks" / "notebook_check_ignores.json"
IGNORELIST_KEY = "private_path_ignored_notebooks"


def load_ignorelist():
    if not IGNORELIST_PATH.exists():
        return set()
    data = json.loads(IGNORELIST_PATH.read_text(encoding="utf-8"))
    return set(data.get(IGNORELIST_KEY, []))


def iter_strings(value, path="root"):
    if isinstance(value, str):
        yield path, value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_strings(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, f"{path}.{key}")


def main():
    failed = False
    ignored = load_ignorelist()

    for notebook_path in sys.argv[1:]:
        rel_obj = Path(notebook_path)
        if rel_obj.parent != Path("."):
            continue
        rel_path = str(rel_obj)
        if rel_path in ignored:
            continue
        try:
            with open(notebook_path, encoding="utf-8") as handle:
                notebook = json.load(handle)
        except Exception as exc:
            print(f"{notebook_path}: failed to read notebook JSON: {exc}", file=sys.stderr)
            failed = True
            continue

        for path, text in iter_strings(notebook):
            for pattern in PATTERNS:
                for match in pattern.finditer(text):
                    snippet = " ".join(text[max(0, match.start() - 30) : match.end() + 30].split())
                    print(
                        f"{notebook_path}: private absolute path found at {path}: {snippet}",
                        file=sys.stderr,
                    )
                    failed = True

    if failed:
        print(
            "Remove private absolute paths from notebook source, metadata, and outputs before committing. "
            f"If this is a known legacy notebook, see {IGNORELIST_PATH.name}.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
