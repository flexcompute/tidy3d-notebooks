#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["nbformat", "pyspellchecker", "nbstripout", "codespell"]
# ///

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import ast
import configparser
import io
import tokenize
from typing import Optional

import nbformat
from spellchecker import SpellChecker


def get_ignore_words_from_config(config_file=".codespellrc") -> list[str]:
    """Reads ignore-words-list from codespell config."""
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        if "codespell" in config and "ignore-words-list" in config["codespell"]:
            words_str = config["codespell"]["ignore-words-list"]
            return [word.strip().lower() for word in words_str.split(",")]
    except Exception:
        pass  # file not found, or parsing error
    return []


def add_words_to_config(new_words: set, config_file=".codespellrc"):
    """Adds words to the ignore-words-list in the codespell config."""
    config = configparser.ConfigParser()
    config.read(config_file)

    if not config.has_section("codespell"):
        config.add_section("codespell")

    if config.has_option("codespell", "ignore-words-list"):
        words_str = config.get("codespell", "ignore-words-list")
        existing_words = {word.strip() for word in words_str.split(",") if word.strip()}
    else:
        existing_words = set()

    word_map = {w.lower(): w for w in existing_words}
    for word in new_words:
        # new_words are already lowercase from pyspellchecker
        word_map[word] = word

    updated_words = sorted(list(word_map.values()), key=str.lower)
    config.set("codespell", "ignore-words-list", ",".join(updated_words))

    with open(config_file, "w") as f:
        config.write(f)

    print(f"\nUpdated {config_file} with {len(new_words)} new word(s).")


def run_interactive_mode(notebooks: list[str]):
    """Runs the spell checker in interactive mode."""
    ignore_words = get_ignore_words_from_config()
    spell = SpellChecker()
    spell.word_frequency.load_words(ignore_words)

    for notebook in notebooks:
        print(f"\nChecking notebook: {notebook}")
        texts, all_identifiers = extract_text_from_notebook(notebook)
        spell.word_frequency.load_words([word.lower() for word in all_identifiers])

        words_to_check = set()
        words_in_parens = set()
        for _, _, text, _ in texts:
            if "/" in text or "\\" in text or "http" in text:
                continue
            words_in_parens.update(w.lower() for w in re.findall(r"\(([a-zA-Z\-']+)\)", text))
            found_words = re.findall(r"\b[a-zA-Z-']+\b", text)
            words_to_check.update(w.lower() for w in found_words if not w.isupper())

        words_to_check -= words_in_parens
        misspelled = spell.unknown(words_to_check)

        if not misspelled:
            print("No spelling errors found.")
            continue

        for word in sorted(list(misspelled)):
            # Find first occurrence for context
            context_line = "No context found."
            for cell_num, line_num, text, source_line in texts:
                if re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
                    context_line = f"Found in Cell {cell_num}, Line {line_num}: {source_line.strip()}"
                    break

            print(f"\nMisspelled word: '{word}'")
            print(context_line)
            answer = input(f"Add '{word}' to dictionary? [Y/n/q] (yes/no/quit) ").lower()

            if answer.strip() == "" or answer == "y":
                add_words_to_config({word})
                spell.word_frequency.add(word)  # update current session for next notebooks
            elif answer == "q":
                print("Quitting interactive session.")
                return
            else:  # 'n'
                print(f"Skipping '{word}'.")

    print("\nInteractive session finished.")


def extract_identifiers_from_code(source: str) -> set[str]:
    """Extracts all identifiers from a python code string."""
    identifiers = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
    except SyntaxError:
        pass  # ignore code that can't be parsed
    return identifiers


def extract_text_from_code(source: str) -> list[tuple[int, str]]:
    """Extracts strings and comments from a python code string."""
    text_nodes = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for toknum, tokval, (srow, _), _, _ in tokens:
            if toknum == tokenize.STRING:
                try:
                    text_nodes.append((srow, ast.literal_eval(tokval)))
                except (ValueError, SyntaxError, MemoryError, TypeError):
                    text_nodes.append((srow, tokval))
            elif toknum == tokenize.COMMENT:
                text_nodes.append((srow, tokval.lstrip("#").strip()))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return text_nodes


def extract_text_from_notebook(notebook_path: str) -> tuple[list[tuple[int, int, str, str]], set[str]]:
    """
    Extracts markdown text, comments, and strings from a notebook.
    Also extracts all python identifiers from code cells.
    Returns a tuple of:
    - list of tuples (cell_num, line_num, text, source_line).
    - set of all python identifiers found.
    """
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except Exception:
        return [], set()

    texts = []
    all_identifiers = set()
    for cell_index, cell in enumerate(notebook.cells):
        cell_num = cell_index + 1
        source_lines = cell.source.splitlines()
        if cell.cell_type == "markdown":
            for line_num, line in enumerate(source_lines):
                texts.append((cell_num, line_num + 1, line, line))
        elif cell.cell_type == "code":
            all_identifiers.update(extract_identifiers_from_code(cell.source))
            code_texts = extract_text_from_code(cell.source)
            for line_num, text in code_texts:
                source_line = (
                    source_lines[line_num - 1] if line_num <= len(source_lines) else ""
                )
                texts.append((cell_num, line_num, text, source_line))

    return texts, all_identifiers


def check_text_against_dictionary(
    texts: list[tuple[int, int, str, str]], ignore_words: list[str], code_identifiers: set[str]
) -> list[str]:
    """
    Checks spelling of words from text.
    Returns list of formatted error strings.
    """
    spell = SpellChecker()
    spell.word_frequency.load_words([word.lower() for word in ignore_words])
    spell.word_frequency.load_words([word.lower() for word in code_identifiers])

    misspelled_lines = []

    for cell_num, line_num, text, source_line in texts:
        if "/" in text or "\\" in text or "http" in text:
            continue

        words_in_parens = {w.lower() for w in re.findall(r"\(([a-zA-Z\-']+)\)", text)}
        words = re.findall(r"\b[a-zA-Z-']+\b", text)
        words_to_check = [
            w for w in words if w.lower() not in words_in_parens and not w.isupper()
        ]

        if not words_to_check:
            continue

        misspelled = spell.unknown(words_to_check)

        if misspelled:
            misspelled_info = ", ".join(f"'{w}'" for w in misspelled)

            misspelled_lines.append(
                f"Cell {cell_num}, Line {line_num}: {misspelled_info}\n  > {source_line.strip()}"
            )

    return misspelled_lines


def get_relative_path(notebook: str) -> str:
    """Get the relative path of the notebook from the current directory."""
    return os.path.relpath(notebook, os.getcwd())


def check_spelling(notebook: str) -> Optional[str]:
    """
    Check spelling in a notebook using both codespell and pyspellchecker.

    Returns:
        A formatted Markdown string containing spelling errors for the notebook,
        using a code block to show codespell's output, or None if no errors were found.
    """
    rel_path = get_relative_path(notebook)
    all_errors = []

    # codespell
    try:
        with open(notebook, encoding="utf-8") as f:
            content = f.read()

        # nbstripout to remove outputs
        nbstripout_proc = subprocess.run(
            ["nbstripout"], input=content, capture_output=True, text=True, check=True
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
            r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "", stripped_content, flags=re.DOTALL
        )

        codespell_proc = subprocess.run(
            ["codespell", "-"],
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
            all_errors.append(f"**{rel_path} (codespell)**:\n```\n{filtered_output}\n```")
    except FileNotFoundError:
        all_errors.append(f"**{rel_path}**: Error - File not found.")
    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(e.cmd)
        all_errors.append(
            f"**{rel_path}**: Error running command `{cmd_str}`:\n```\n{e.stderr}\n```"
        )
    except Exception as e:
        all_errors.append(
            f"**{rel_path}**: An unexpected error occurred with codespell:\n```\n{str(e)}\n```"
        )

    # pyspellchecker
    try:
        ignore_words = get_ignore_words_from_config()
        texts, all_identifiers = extract_text_from_notebook(notebook)
        pyspell_errors = check_text_against_dictionary(texts, ignore_words, all_identifiers)
        if pyspell_errors:
            error_details = "\n".join(pyspell_errors)
            all_errors.append(f"**{rel_path} (pyspellchecker)**:\n```\n{error_details}\n```")
    except Exception as e:
        all_errors.append(
            f"**{rel_path}**: An unexpected error with pyspellchecker:\n```\n{str(e)}\n```"
        )

    if all_errors:
        return "\n\n".join(all_errors)
    return None


def main():
    parser = argparse.ArgumentParser(description="Check spelling in Jupyter notebooks")
    parser.add_argument("notebooks", nargs="+", help="List of notebook files to check")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode to add words to the dictionary.",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode(args.notebooks)
        sys.exit(0)

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
