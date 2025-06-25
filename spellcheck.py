#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["nbformat", "pyspellchecker"]
# ///

import argparse
import concurrent.futures
import os
import re
import sys
import ast
import io
import logging
import tokenize
from collections import Counter
from typing import Optional

import nbformat
from spellchecker import SpellChecker


CUSTOM_DICT_PATH = "custom_dictionary.json"


def load_custom_words(spell: SpellChecker):
    """Loads words from a custom dictionary file into the SpellChecker instance."""
    if os.path.exists(CUSTOM_DICT_PATH):
        spell.word_frequency.load_dictionary(CUSTOM_DICT_PATH)


def add_words_to_custom_dictionary(new_words: set[str]):
    """Adds words to the custom dictionary file."""
    temp_spell = SpellChecker(language=None)
    if os.path.exists(CUSTOM_DICT_PATH):
        temp_spell.word_frequency.load_dictionary(CUSTOM_DICT_PATH)

    for word in new_words:
        temp_spell.word_frequency.add(word)

    temp_spell.export(CUSTOM_DICT_PATH, gzipped=False)
    print(f"\nUpdated {CUSTOM_DICT_PATH} with {len(new_words)} new word(s).")


def run_interactive_mode(notebooks: list[str], reference_words: set[str]):
    """Runs the spell checker in interactive mode."""
    spell = SpellChecker()
    load_custom_words(spell)
    spell.word_frequency.load_words(reference_words)

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
                    context_line = (
                        f"Found in Cell {cell_num}, Line {line_num}: {source_line.strip()}"
                    )
                    break

            print(f"\nMisspelled word: '{word}'")
            print(context_line)
            answer = input(f"Add '{word}' to dictionary? [Y/n/q] (yes/no/quit) ").lower()

            if answer.strip() == "" or answer == "y":
                add_words_to_custom_dictionary({word})
                spell.word_frequency.add(word)  # update current session for next notebooks
            elif answer == "q":
                print("Quitting interactive session.")
                return
            else:  # 'n'
                print(f"Skipping '{word}'.")

    print("\nInteractive session finished.")


def extract_identifiers_from_code(source: str) -> set[str]:
    """Extracts and splits identifiers from a python code string."""
    identifiers = set()
    raw_identifiers = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                raw_identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                raw_identifiers.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raw_identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                raw_identifiers.add(node.arg)
    except SyntaxError:
        pass  # ignore code that can't be parsed

    for identifier in raw_identifiers:
        # Split by snake_case and camelCase
        words = re.sub(r"_", " ", identifier)
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", words)
        identifiers.update(word.lower() for word in words.split() if word)

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


def extract_text_from_notebook(
    notebook_path: str,
) -> tuple[list[tuple[int, int, str, str]], set[str]]:
    """
    Extracts markdown text, comments, and strings from a notebook.
    Also extracts all python identifiers from code cells.
    Returns a tuple of:
    - list of tuples (cell_num, line_num, text, source_line).
    - set of all python identifiers found.
    """
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except Exception as e:
        logging.warning(f"Could not read or parse notebook '{notebook_path}': {e}")
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
                source_line = source_lines[line_num - 1] if line_num <= len(source_lines) else ""
                texts.append((cell_num, line_num, text, source_line))

    return texts, all_identifiers


def get_all_words_from_notebook(notebook_path: str) -> set[str]:
    """Extracts all unique words from a notebook."""
    words = set()
    texts, identifiers = extract_text_from_notebook(notebook_path)
    words.update(identifiers)

    for _, _, text, _ in texts:
        if "/" in text or "\\" in text or "http" in text:
            continue
        found_words = re.findall(r"\b[a-zA-Z-']+\b", text)
        words.update(w.lower() for w in found_words)
    return words


def build_reference_word_set(reference_notebooks: list[str], threshold: int) -> set[str]:
    """Builds a set of words that appear in at least 'threshold' reference notebooks."""
    if not reference_notebooks or threshold <= 0:
        return set()

    logging.info(f"Building reference dictionary from {len(reference_notebooks)} notebooks...")

    word_counts = Counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_notebook = {
            executor.submit(get_all_words_from_notebook, nb): nb for nb in reference_notebooks
        }
        for future in concurrent.futures.as_completed(future_to_notebook):
            try:
                words_in_notebook = future.result()
                # Each word is counted once per notebook
                word_counts.update(words_in_notebook)
            except Exception as exc:
                notebook = future_to_notebook[future]
                logging.warning(f"Could not process reference notebook {notebook}: {exc}")

    reference_words = {word for word, count in word_counts.items() if count >= threshold}
    logging.info(f"Found {len(reference_words)} words meeting the threshold of {threshold}.")
    return reference_words


def check_text_against_dictionary(
    texts: list[tuple[int, int, str, str]],
    code_identifiers: set[str],
    reference_words: set[str],
) -> list[str]:
    """
    Checks spelling of words from text.
    Returns list of formatted error strings.
    """
    spell = SpellChecker()
    load_custom_words(spell)
    spell.word_frequency.load_words([word.lower() for word in code_identifiers])
    spell.word_frequency.load_words(reference_words)

    misspelled_lines = []

    for cell_num, line_num, text, source_line in texts:
        if "/" in text or "\\" in text or "http" in text:
            continue

        words_in_parens = {w.lower() for w in re.findall(r"\(([a-zA-Z\-']+)\)", text)}
        words = re.findall(r"\b[a-zA-Z-']+\b", text)
        words_to_check = [w for w in words if w.lower() not in words_in_parens and not w.isupper()]

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


def check_spelling(notebook: str, reference_words: set[str]) -> Optional[str]:
    """
    Check spelling in a notebook using pyspellchecker.

    Returns:
        A formatted Markdown string containing spelling errors for the notebook,
        or None if no errors were found.
    """
    rel_path = get_relative_path(notebook)

    try:
        texts, all_identifiers = extract_text_from_notebook(notebook)
        pyspell_errors = check_text_against_dictionary(
            texts, all_identifiers, reference_words
        )
        if pyspell_errors:
            error_details = "\n".join(pyspell_errors)
            return f"**{rel_path}**:\n```\n{error_details}\n```"
    except Exception as e:
        logging.error(f"An unexpected error processing notebook {rel_path}", exc_info=True)
        return f"**{rel_path}**: An unexpected error with pyspellchecker:\n```\n{str(e)}\n```"

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
    parser.add_argument(
        "--reference-notebooks",
        nargs="+",
        default=None,
        help="Reference notebooks to build a dictionary of common words. If not provided, all other notebooks in the current directory are used.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Minimum number of occurrences in reference notebooks for a word to be ignored.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging output."
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    reference_notebooks = args.reference_notebooks
    if reference_notebooks is None:
        logging.info(
            "No reference notebooks provided, using all other notebooks in the current directory as reference."
        )
        all_notebooks_in_dir = [f for f in os.listdir(".") if f.endswith(".ipynb")]
        notebooks_to_check_set = set(args.notebooks)
        reference_notebooks = [
            nb for nb in all_notebooks_in_dir if nb not in notebooks_to_check_set
        ]

    reference_words = build_reference_word_set(reference_notebooks, args.threshold)

    if args.interactive:
        run_interactive_mode(args.notebooks, reference_words)
        sys.exit(0)

    all_errors: list[str] = []
    num_files_processed = 0
    num_files_with_errors = 0
    num_files_with_processing_errors = 0

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for notebook in args.notebooks:
            futures.append(executor.submit(check_spelling, notebook, reference_words))

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
                logging.error("An unexpected error occurred processing a task", exc_info=True)
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
