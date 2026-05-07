#!/usr/bin/env python3

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ("title", "description", "keywords", "feature_image")
CASE_STUDY_FIELDS = ("applications", "features")
REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORELIST_PATH = REPO_ROOT / "checks" / "notebook_check_ignores.json"
IGNORELIST_KEY = "metadata_validation_ignored_notebooks"


def load_ignorelist() -> set[str]:
    if not IGNORELIST_PATH.exists():
        return set()
    data = json.loads(IGNORELIST_PATH.read_text(encoding="utf-8"))
    return set(data.get(IGNORELIST_KEY, []))


def validate_notebook(path_str: str) -> list[str]:
    path = Path(path_str)
    errors: list[str] = []

    if not path.exists():
        return []

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: failed to read notebook JSON: {exc}"]

    if not isinstance(notebook, dict):
        return [f"{path}: notebook root must be a JSON object"]

    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        return [f"{path}: metadata must be a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in metadata:
            errors.append(f"{path}: missing metadata.{field}")

    for field in REQUIRED_FIELDS:
        if field not in metadata:
            continue
        value = metadata[field]
        if not isinstance(value, str):
            errors.append(f"{path}: metadata.{field} must be a string")
            continue
        if field != "feature_image" and not value.strip():
            errors.append(f"{path}: metadata.{field} must not be empty")

    feature_image = metadata.get("feature_image")
    if isinstance(feature_image, str) and feature_image:
        if not feature_image.startswith("./img/"):
            errors.append(f"{path}: metadata.feature_image must point under ./img/")
        image_path = (path.parent / feature_image).resolve()
        if not image_path.exists():
            errors.append(f"{path}: metadata.feature_image does not exist: {feature_image}")

    has_case_study_metadata = any(field in metadata for field in CASE_STUDY_FIELDS)
    if has_case_study_metadata:
        for field in CASE_STUDY_FIELDS:
            if field not in metadata:
                errors.append(f"{path}: metadata.{field} is required when using case-study tags")
                continue
            value = metadata[field]
            if not isinstance(value, list):
                errors.append(f"{path}: metadata.{field} must be a list of strings")
                continue
            if any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append(f"{path}: metadata.{field} must contain only non-empty strings")

    return errors


def main() -> int:
    all_errors: list[str] = []
    ignored = load_ignorelist()
    for notebook_path in sys.argv[1:]:
        rel_obj = Path(notebook_path)
        if rel_obj.parent != Path("."):
            continue
        if notebook_path.endswith(".ipynb") and notebook_path not in ignored:
            all_errors.extend(validate_notebook(notebook_path))

    if all_errors:
        all_errors.append(
            f"Known legacy notebooks can be listed in {IGNORELIST_PATH.name} under {IGNORELIST_KEY}."
        )
        print("\n".join(all_errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
