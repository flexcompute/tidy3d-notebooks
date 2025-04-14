#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from typing import Optional


def get_relative_path(notebook: str) -> str:
    """Get the relative path of the notebook from the current directory."""
    return os.path.relpath(notebook, os.getcwd())


def check_spelling(notebook: str) -> Optional[str]:
    """
    Check spelling in a notebook.

    Returns:
        A formatted Markdown string containing spelling errors for the notebook,
        using a code block to show codespell's output, or None if no errors were found.
    """
    rel_path = get_relative_path(notebook)
    error_message_block = None

    try:
        with open(notebook, encoding="utf-8") as f:
            content = f.read()

        # nbstripout to remove outputs
        nbstripout_proc = subprocess.run(
            ["uvx", "nbstripout"],
            input=content,
            capture_output=True,
            text=True,
            check=True,
        )

        # remove image tags with base64 data
        stripped_content = re.sub(
            r'<img\s+src="data:image/[^"]+;base64,[^"]+"[^>]*>|<img\s+src="data:image/[^"]+;base64,[^"]+"[^/>]*/>',
            "",
            nbstripout_proc.stdout,
            flags=re.DOTALL,
        )

        # remove any remaining base64 strings that might appear without proper HTML tags
        stripped_content = re.sub(
            r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
            "",
            stripped_content,
            flags=re.DOTALL,
        )

        codespell_proc = subprocess.run(
            ["uvx", "codespell", "-"],
            input=stripped_content,
            capture_output=True,
            text=True,
            check=False,  # codespell exits non-zero on errors, which is expected
        )

        # filter codespell's config file lines
        output_lines = []
        for line in codespell_proc.stdout.splitlines():
            if line.strip().startswith("Used config files:") or re.match(
                r"^\s+\d+:\s+\.codespellrc", line
            ):
                continue
            output_lines.append(line.replace("-:", "Line ", 1))

        filtered_output = "\n".join(output_lines).strip()

        if filtered_output:
            error_message_block = f"**{rel_path}**:\n```\n{filtered_output}\n```"

    except FileNotFoundError:
        error_message_block = f"**{rel_path}**: Error - File not found."
    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(e.cmd)
        error_message_block = (
            f"**{rel_path}**: Error running command `{cmd_str}`:\n```\n{e.stderr}\n```"
        )
    except Exception as e:
        error_message_block = f"**{rel_path}**: An unexpected error occurred:\n```\n{str(e)}\n```"

    return error_message_block


def main():
    parser = argparse.ArgumentParser(description="Check spelling in Jupyter notebooks")
    parser.add_argument("notebooks", nargs="+", help="List of notebook files to check")
    args = parser.parse_args()

    all_errors: list[str] = []
    num_files_processed = 0
    num_files_with_errors = 0
    num_files_with_processing_errors = 0

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for notebook in args.notebooks:
            futures.append(executor.submit(check_spelling, notebook))

        for future in concurrent.futures.as_completed(futures):
            num_files_processed += 1
            try:
                error_output = future.result()
                if error_output:
                    all_errors.append(error_output)
                    if (
                        "Error running command" in error_output
                        or "An unexpected error occurred" in error_output
                        or "Error - File not found" in error_output
                    ):
                        num_files_with_processing_errors += 1
                    else:
                        num_files_with_errors += 1
            except Exception as exc:
                print(f"An unexpected error occurred processing a task: {exc}", file=sys.stderr)
                num_files_with_processing_errors += 1
                all_errors.append(
                    f"**Unknown File**: An unexpected error occurred during processing:\n```\n{exc}\n```"
                )

    if all_errors:
        all_errors.sort()

        print("## Spell Check Report\n")
        print("\n\n---\n\n".join(all_errors))

        summary_lines = []
        if num_files_with_errors > 0:
            summary_lines.append(f"Found spelling errors in {num_files_with_errors} file(s).")
        if num_files_with_processing_errors > 0:
            summary_lines.append(
                f"Encountered processing errors in {num_files_with_processing_errors} file(s)."
            )
        if not summary_lines and all_errors:
            summary_lines.append(f"Found issues in {len(all_errors)} file(s).")

        total_notebooks_input = len(args.notebooks)
        print(f"\n---\nChecked {total_notebooks_input} notebook(s). " + " ".join(summary_lines))
        sys.exit(1)
    else:
        total_notebooks_input = len(args.notebooks)
        print(f"Spell check passed successfully for {total_notebooks_input} notebook(s).")
        sys.exit(0)


if __name__ == "__main__":
    main()
