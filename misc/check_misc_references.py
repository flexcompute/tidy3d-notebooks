#!/usr/bin/env python3
"""
Scan all ipynb files in the root directory and check if any file references
misc/ directory files that either don't exist or are not listed in
misc/import_file_mapping.json.

Usage:
    python check_misc_references.py          # Check only
    python check_misc_references.py --fix    # Check and auto-fix missing mappings
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Methods that write files to misc directory (these references should be ignored)
WRITE_METHODS = [
    "write_gds",
    "to_gds_file",
    "to_file",
]

# Variable name patterns that indicate output/write file paths
# These are checked as substrings in variable names before "="
WRITE_VAR_PATTERNS = [
    "history_fname",
    "history_file_path",
]


def get_project_root() -> Path:
    """Get the project root directory (parent of misc)."""
    return Path(__file__).parent.parent


def load_import_file_mapping(misc_dir: Path) -> dict[str, list[str]]:
    """Load import_file_mapping.json."""
    mapping_file = misc_dir / "import_file_mapping.json"
    if not mapping_file.exists():
        print(f"Error: Cannot find {mapping_file}")
        sys.exit(1)
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_misc_files(misc_dir: Path) -> set[str]:
    """Get all file names in the misc directory."""
    return {f.name for f in misc_dir.iterdir() if f.is_file()}


def is_write_context(content: str, match_start: int) -> bool:
    """
    Check if the match position is in a write method or write variable context.
    Look backwards from the match position to see if it's preceded by a write method
    or a variable assignment with a write-related variable name.
    """
    # Look at the 200 characters before the match
    lookback = content[max(0, match_start - 200) : match_start]

    # Check if any write method appears in the lookback context
    for method in WRITE_METHODS:
        # Match patterns like ".write_gds(" or "write_gds("
        if re.search(rf"\.?{re.escape(method)}\s*\([^)]*$", lookback):
            return True

    # Check if this is a variable assignment with a write-related variable name
    # Match patterns like "history_fname = " or "history_fname="
    for var_pattern in WRITE_VAR_PATTERNS:
        if re.search(rf"{re.escape(var_pattern)}\s*=\s*$", lookback):
            return True

    return False


def find_misc_references(notebook_path: Path) -> set[str]:
    """
    Find all references to misc/ directory in an ipynb file.
    Returns a set of referenced file names (without misc/ prefix).
    Excludes references that appear in write method contexts.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ipynb files are JSON format, double quotes in strings are escaped as \"
    # Match patterns like "misc/xxx", 'misc/xxx', \"misc/xxx\", "./misc/xxx", etc.
    # File names contain only valid characters: letters, numbers, underscores, hyphens, dots
    patterns = [
        # Match escaped quotes \"./misc/xxx\" or \"misc/xxx\"
        r'\\\"(?:\./)?misc/([a-zA-Z0-9_\-\.]+)\\\"',
        # Match regular quotes "./misc/xxx" or "misc/xxx" or './misc/xxx'
        r'["\'](?:\./)?misc/([a-zA-Z0-9_\-\.]+)["\']',
    ]

    references = set()
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            filename = match.group(1).strip()

            # Filter out invalid file names
            if not filename:
                continue
            # Skip files starting with . (hidden files)
            if filename.startswith("."):
                continue
            # Skip files ending with _ (likely part of string concatenation)
            if filename.endswith("_"):
                continue
            # Skip if this is in a write method context (output file, not input)
            if is_write_context(content, match.start()):
                continue

            references.add(filename)

    return references


def check_notebook(
    notebook_path: Path,
    misc_files: set[str],
    import_mapping: dict[str, list[str]],
    errors: list[str],
    missing_mappings: dict[str, list[str]],
) -> None:
    """Check a single notebook file."""
    notebook_name = notebook_path.name
    references = find_misc_references(notebook_path)

    if not references:
        return

    # Get the files declared for this notebook in the mapping
    declared_files = set(import_mapping.get(notebook_name, []))

    for ref in references:
        # Check if the file exists in the misc directory
        if ref not in misc_files:
            errors.append(
                f"[FILE NOT FOUND] {notebook_name}: references 'misc/{ref}', "
                f"but the file does not exist in misc directory"
            )
            continue

        # Check if the file is declared in import_file_mapping.json
        if ref not in declared_files:
            errors.append(
                f"[NOT IN MAPPING] {notebook_name}: references 'misc/{ref}', "
                f"but it is not declared in import_file_mapping.json"
            )
            # Collect missing mappings for potential fix
            if notebook_name not in missing_mappings:
                missing_mappings[notebook_name] = []
            if ref not in missing_mappings[notebook_name]:
                missing_mappings[notebook_name].append(ref)


def update_import_file_mapping(
    misc_dir: Path,
    import_mapping: dict[str, list[str]],
    missing_mappings: dict[str, list[str]],
) -> None:
    """Update import_file_mapping.json with missing mappings."""
    # Merge missing mappings into import_mapping
    for notebook_name, files in missing_mappings.items():
        if notebook_name in import_mapping:
            # Add new files to existing entry
            existing = set(import_mapping[notebook_name])
            for f in files:
                if f not in existing:
                    import_mapping[notebook_name].append(f)
        else:
            # Create new entry
            import_mapping[notebook_name] = files

    # Write back to file in original one-line-per-entry format
    mapping_file = misc_dir / "import_file_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(import_mapping.items())
        for i, (notebook_name, files) in enumerate(items):
            # Format: "notebook.ipynb": [ "file1.ext",  "file2.ext"]
            files_str = ", ".join(f' "{file}"' for file in files)
            line = f'    "{notebook_name}": [{files_str}]'
            if i < len(items) - 1:
                line += ","
            f.write(line + "\n")
        f.write("}\n")

    print(f"\nUpdated {mapping_file}")
    print(f"Added mappings for {len(missing_mappings)} notebook(s)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check misc directory references in notebooks."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix missing mappings by adding them to import_file_mapping.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = get_project_root()
    misc_dir = root_dir / "misc"

    print(f"Project root: {root_dir}")
    print(f"Misc directory: {misc_dir}")
    print("-" * 60)

    # Load data
    import_mapping = load_import_file_mapping(misc_dir)
    misc_files = get_misc_files(misc_dir)

    print(f"Found {len(misc_files)} files in misc directory")
    print(f"Found {len(import_mapping)} notebook mappings in import_file_mapping.json")
    print("-" * 60)

    # Scan all ipynb files
    notebooks = list(root_dir.glob("*.ipynb"))
    print(f"Found {len(notebooks)} notebook files")
    print("-" * 60)

    errors = []
    missing_mappings: dict[str, list[str]] = {}
    for notebook_path in sorted(notebooks):
        check_notebook(notebook_path, misc_files, import_mapping, errors, missing_mappings)

    # Output results
    if errors:
        print(f"\nFound {len(errors)} issues:\n")
        for error in errors:
            print(f"  X {error}")
        print()

        # Auto-fix if requested
        if args.fix and missing_mappings:
            update_import_file_mapping(misc_dir, import_mapping, missing_mappings)

        # Only return error if there are FILE NOT FOUND errors
        # (missing mappings are fixed if --fix is used)
        if args.fix:
            file_not_found_errors = [e for e in errors if "[FILE NOT FOUND]" in e]
            if file_not_found_errors:
                sys.exit(1)
            else:
                print("\n[OK] Missing mappings have been fixed!\n")
                sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("\n[OK] All checks passed, no issues found!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
