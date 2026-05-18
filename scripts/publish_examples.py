#!/usr/bin/env python3
"""
Publish examples to CDN directory.

Usage:
    python scripts/publish_examples.py <version>

Example:
    python scripts/publish_examples.py 2.10.0

This script performs the following operations:
1. Run misc/check_misc_references.py to validate references, exit if failed
2. Run scripts/examples_config_gen.py to generate config.yaml
3. Create target folder ../cdn.simulation.cloud/notebook/examples-{version}
4. Copy all ipynb files and config.yaml from root to target folder
5. Copy misc files listed in import_file_mapping.json to target misc folder
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[A-Za-z0-9][A-Za-z0-9._+-]*)?$")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]


def validate_version(version: str) -> str:
    """Validate the user-provided release version used in the target folder name."""
    if (
        not VERSION_PATTERN.fullmatch(version)
        or ".." in version
        or "/" in version
        or "\\" in version
    ):
        raise ValueError(
            "Version must look like a release version, for example '2.10.0' or '2.10.0rc1'."
        )
    return version


def run_check_misc_references(root_dir: Path) -> bool:
    """Run misc/check_misc_references.py script.

    Returns:
        True if check passed, False if check failed
    """
    script_path = root_dir / "misc" / "check_misc_references.py"
    print(f"Running: {script_path}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=root_dir,
    )

    if result.returncode != 0:
        print("\n[ERROR] misc/check_misc_references.py failed, aborting publish.")
        return False

    return True


def run_examples_config_gen(root_dir: Path) -> bool:
    """Run scripts/examples_config_gen.py script to generate config.yaml.

    Returns:
        True if successful, False if failed
    """
    script_path = root_dir / "scripts" / "examples_config_gen.py"
    print(f"\nRunning: {script_path}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=root_dir,
    )

    if result.returncode != 0:
        print("\n[ERROR] scripts/examples_config_gen.py failed.")
        return False

    config_file = root_dir / "config.yaml"
    if not config_file.exists():
        print("\n[ERROR] config.yaml was not generated.")
        return False

    print(f"[OK] config.yaml generated: {config_file}")
    return True


def prepare_target_directory(root_dir: Path, version: str) -> Path:
    """Prepare target directory, clear if exists.

    Returns:
        Path to the target directory
    """
    version = validate_version(version)
    base_dir = (root_dir.parent / "cdn.simulation.cloud" / "notebook").resolve(strict=False)
    target_dir = (base_dir / f"examples-{version}").resolve(strict=False)
    try:
        target_dir.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(
            f"Target directory escapes the notebook CDN directory: {target_dir}"
        ) from exc

    print(f"\nTarget directory: {target_dir}")
    print("-" * 60)

    if target_dir.exists():
        print("Target directory exists, clearing...")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Target directory created")
    return target_dir


def copy_notebooks_and_config(root_dir: Path, target_dir: Path) -> int:
    """Copy ipynb files and config.yaml to target directory.

    Returns:
        Number of files copied
    """
    print("\nCopying notebook and config.yaml files...")
    print("-" * 60)

    # Copy all ipynb files
    ipynb_files = list(root_dir.glob("*.ipynb"))
    count = 0

    for ipynb_file in ipynb_files:
        dest = target_dir / ipynb_file.name
        shutil.copy2(ipynb_file, dest)
        count += 1

    print(f"[OK] Copied {count} ipynb files")

    # Copy config.yaml
    config_file = root_dir / "config.yaml"
    if config_file.exists():
        shutil.copy2(config_file, target_dir / "config.yaml")
        count += 1
        print("[OK] Copied config.yaml")
    else:
        print("[WARNING] config.yaml does not exist")

    return count


def load_import_file_mapping(root_dir: Path) -> dict[str, list[str]]:
    """Load import_file_mapping.json file."""
    mapping_file = root_dir / "misc" / "import_file_mapping.json"
    if not mapping_file.exists():
        print(f"[ERROR] Cannot find {mapping_file}")
        return {}

    with open(mapping_file, encoding="utf-8") as f:
        return json.load(f)


def resolve_mapping_path(base_dir: Path, filename: str) -> Path:
    """Resolve a mapped file path and ensure it stays inside base_dir."""
    base_dir = base_dir.resolve(strict=False)
    path = (base_dir / filename).resolve(strict=False)
    try:
        path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Mapped misc file path escapes its base directory: {filename}") from exc
    return path


def copy_misc_files(root_dir: Path, target_dir: Path) -> int:
    """Copy misc files listed in import_file_mapping.json.

    Returns:
        Number of files copied
    """
    print("\nCopying misc data files...")
    print("-" * 60)

    import_mapping = load_import_file_mapping(root_dir)
    if not import_mapping:
        return 0

    # Collect all files to copy (deduplicated)
    files_to_copy: set[str] = set()
    for files in import_mapping.values():
        files_to_copy.update(files)

    # Create target misc directory
    target_misc_dir = target_dir / "misc"
    target_misc_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    misc_dir = root_dir / "misc"
    count = 0
    missing_files = []

    for filename in sorted(files_to_copy):
        src = resolve_mapping_path(misc_dir, filename)
        dest = resolve_mapping_path(target_misc_dir, filename)
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            count += 1
        else:
            missing_files.append(filename)

    print(f"[OK] Copied {count} misc files")

    if missing_files:
        print(
            "[WARNING] The following files are listed in import_file_mapping.json but not found in misc directory:"
        )
        for f in missing_files:
            print(f"  - {f}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Publish examples to CDN directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/publish_examples.py 2.10.0
        """,
    )
    parser.add_argument(
        "version",
        help="Version number, e.g. 2.10.0",
    )
    args = parser.parse_args()

    try:
        version = validate_version(args.version)
    except ValueError as exc:
        parser.error(str(exc))

    print("=" * 60)
    print(f"Publishing Examples version: {version}")
    print("=" * 60)

    root_dir = get_project_root()
    print(f"Project root: {root_dir}")

    # Step 1: Run check_misc_references.py
    if not run_check_misc_references(root_dir):
        sys.exit(1)

    # Step 2: Run scripts/examples_config_gen.py
    if not run_examples_config_gen(root_dir):
        sys.exit(1)

    # Step 3: Prepare target directory
    target_dir = prepare_target_directory(root_dir, version)

    # Step 4: Copy ipynb and config.yaml files
    notebook_count = copy_notebooks_and_config(root_dir, target_dir)

    # Step 5: Copy misc files
    misc_count = copy_misc_files(root_dir, target_dir)

    # Done
    print("\n" + "=" * 60)
    print("[DONE] Publish successful!")
    print(f"  - Target directory: {target_dir}")
    print(f"  - Notebook files: {notebook_count}")
    print(f"  - Misc data files: {misc_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
