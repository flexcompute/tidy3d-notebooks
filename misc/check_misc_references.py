#!/usr/bin/env python3
"""
Scan all ipynb files in the root directory and check if any file references
misc/ directory files that either don't exist or are not listed in
misc/import_file_mapping.json.

Uses AST (Abstract Syntax Tree) analysis to accurately determine whether
file references are for reading or writing operations.

Usage:
    python check_misc_references.py              # Check only
    python check_misc_references.py --fix        # Check and auto-fix missing mappings
    python check_misc_references.py --analyze    # Show detailed usage analysis
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# =============================================================================
# USAGE TYPE CLASSIFICATION
# =============================================================================


class UsageType(Enum):
    """Type of file usage."""

    WRITE = "WRITE"
    READ = "READ"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# CUSTOMIZABLE CONFIGURATION - Extend these lists as needed
# =============================================================================

# Method/function names that indicate WRITE operations
WRITE_METHODS = {
    # Tidy3D export methods
    "write_gds",
    "to_gds_file",
    "to_file",
    "to_hdf5",
    "to_json",
    "save",
    # Pickle/JSON
    "dump",  # pickle.dump, json.dump
    # NumPy save
    "savetxt",
    "savez",
    "savez_compressed",
    # Pandas save
    "to_csv",
    "to_excel",
    "to_pickle",
    "to_parquet",
    "to_feather",
    "to_hdf",
}

# Method/function names that indicate READ operations
READ_METHODS = {
    # Tidy3D import methods
    "from_file",
    "from_gds",
    "from_stl",
    "from_hdf5",
    "from_json",
    "from_vtu",
    "load",
    # GDStk/GDS read
    "read_gds",
    # NumPy read
    "loadtxt",
    "genfromtxt",
    # Pandas read
    "read_csv",
    "read_excel",
    "read_pickle",
    "read_parquet",
    "read_json",
    "read_feather",
    "read_hdf",
    # PIL/Image
    "imread",
    # Tidy3D KLayout plugin
    "DRCRunner",
}

# Files that are allowed to not exist in misc directory
# These are typically intermediate/output files that may or may not be present
OPTIONAL_FILES = {
    # Optimization history files (generated during notebook execution)
    "optimization_history_FAID_4ch_R100_1filter_buffer.pkl",
}

# Variable name patterns that indicate optional files
# Files referenced through these variable names are allowed to not exist
# (e.g., history files that try to load previous run results)
OPTIONAL_VAR_PATTERNS = {
    "history_fname",
    "history_file",
    "history_path",
}

# Raw fallback for code cells that contain notebook magics or shell syntax and
# therefore cannot be parsed by ast.parse().
RAW_MISC_PATH_PATTERN = re.compile(
    r"""(?P<quote>["'])(?:\./)?misc/(?P<filename>[^"'\s)]+)(?P=quote)"""
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FileReference:
    """A reference to a misc/ file in a notebook."""

    notebook: str
    filename: str
    line_number: int
    context: str
    usage_type: UsageType = UsageType.UNKNOWN
    detection_reason: str = ""
    var_name: str = ""  # Variable name if file was assigned to a variable


@dataclass
class AnalysisResult:
    """Result of analyzing all notebooks."""

    references: list[FileReference] = field(default_factory=list)

    def _files_by_type(self, usage_type: UsageType) -> dict[str, list[str]]:
        """Group filenames by notebook for a given usage type."""
        result: dict[str, list[str]] = {}
        for ref in self.references:
            if ref.usage_type == usage_type:
                result.setdefault(ref.notebook, []).append(ref.filename)
        return result

    @property
    def write_files(self) -> dict[str, list[str]]:
        return self._files_by_type(UsageType.WRITE)

    @property
    def read_files(self) -> dict[str, list[str]]:
        return self._files_by_type(UsageType.READ)

    @property
    def unknown_files(self) -> dict[str, list[str]]:
        return self._files_by_type(UsageType.UNKNOWN)


@dataclass
class FunctionInfo:
    """Information about a function definition."""

    name: str
    params: list[str]  # Parameter names
    # Map: param_name -> [(callee_func, callee_param_name), ...]
    param_flows: dict[str, list[tuple[str, str]]] = field(default_factory=dict)


@dataclass(frozen=True)
class NotebookCell:
    """A single code cell extracted from a notebook (with a parsed AST if valid)."""

    index: int
    source_code: str
    source_lines: list[str]
    tree: ast.AST | None


@dataclass(frozen=True)
class Assignment:
    """A variable assignment involving a misc file."""

    var_name: str
    filename: str
    cell_index: int
    line_number: int
    context: str


@dataclass(frozen=True)
class VariableUsage:
    """A usage of a tracked variable."""

    var_name: str
    cell_index: int
    line_number: int
    usage_type: UsageType
    reason: str


@dataclass(frozen=True)
class DirectReference:
    """A direct string reference to a misc file."""

    filename: str
    cell_index: int
    line_number: int
    context: str
    usage_type: UsageType
    reason: str


# =============================================================================
# AST ANALYSIS HELPERS
# =============================================================================


def get_project_root() -> Path:
    """Get the project root directory (parent of misc)."""
    return Path(__file__).parent.parent


def extract_notebook_cells(notebook_path: Path) -> list[NotebookCell]:
    """
    Extract code cells from a notebook.
    Returns list of NotebookCell with parsed AST (tree=None if SyntaxError).
    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells: list[NotebookCell] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_lines = [line.rstrip("\n") for line in source]
                source_code = "\n".join(source_lines)
            else:
                source_code = source
                source_lines = source.split("\n")

            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                tree = None

            cells.append(
                NotebookCell(
                    index=idx,
                    source_code=source_code,
                    source_lines=source_lines,
                    tree=tree,
                )
            )
    return cells


def try_eval_string_expr(node: ast.AST, constants: dict[str, str] | None = None) -> str | None:
    """
    Try to evaluate a string expression (including concatenation).
    Returns the evaluated string if successful, None otherwise.

    Supports:
    - String literals: "misc/file.txt"
    - String concatenation: "misc/" + "file.txt"
    - f-strings: f"misc/{var}.txt"
    - Path division: Path("misc") / "file.txt"
    - os.path.join: join("misc", "file.txt")
    """
    if constants is None:
        constants = {}

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]

    # String concatenation with +
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = try_eval_string_expr(node.left, constants)
        right = try_eval_string_expr(node.right, constants)
        if left is not None and right is not None:
            return left + right

    # Path division: Path("misc") / "file.txt"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = try_eval_string_expr(node.left, constants)
        right = try_eval_string_expr(node.right, constants)
        if left is not None and right is not None:
            # Join with "/" like pathlib.Path does
            return f"{left.rstrip('/')}/{right.lstrip('/')}"

    # Handle Path("misc") constructor
    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Path("misc") / "file.txt" or Path("misc", "file.txt")
        if func_name in ("Path", "PurePath", "PosixPath", "WindowsPath"):
            if node.args:
                if len(node.args) == 1:
                    return try_eval_string_expr(node.args[0], constants)
                parts = []
                for arg in node.args:
                    part = try_eval_string_expr(arg, constants)
                    if part is None:
                        return None
                    parts.append(part)
                return join_path_parts(parts)

        # os.path.join("misc", "file.txt") or join("misc", "file.txt")
        if func_name == "join":
            parts = []
            for arg in node.args:
                part = try_eval_string_expr(arg, constants)
                if part is None:
                    return None
                parts.append(part)
            if parts:
                return join_path_parts(parts)

    # Handle JoinedStr (f-strings) - basic support
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                result = try_eval_string_expr(value.value, constants)
                if result is not None:
                    parts.append(result)
                else:
                    return None
            else:
                return None
        return "".join(parts)

    return None


def join_path_parts(parts: list[str]) -> str:
    """Join path fragments while preserving a trailing slash on directory-like paths."""
    cleaned_parts = []
    for idx, part in enumerate(parts):
        if idx == 0:
            cleaned_parts.append(part.rstrip("/"))
        elif idx == len(parts) - 1:
            cleaned_parts.append(part.lstrip("/"))
        else:
            cleaned_parts.append(part.strip("/"))
    return "/".join(cleaned_parts)


def extract_misc_filename(value: str) -> str | None:
    """
    Extract the filename from a misc/ path string.
    Returns the filename if valid, None otherwise.
    """
    if "misc/" not in value:
        return None

    # Get the part after the last "misc/"
    after_misc = value.split("misc/")[-1]

    # Filter out: empty, directory paths, hidden files
    if not after_misc or after_misc.endswith("/") or after_misc.startswith("."):
        return None

    return after_misc


def is_misc_path(node: ast.AST, constants: dict[str, str] | None = None) -> str | None:
    """
    Check if an AST node is a string containing a misc/ file path.
    Returns the filename if it is, None otherwise.
    """
    value = try_eval_string_expr(node, constants)
    if value is not None:
        return extract_misc_filename(value)
    return None


def get_open_mode(call_node: ast.Call) -> str:
    """Extract the mode argument from an open() call."""
    if len(call_node.args) >= 2:
        mode_arg = call_node.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            return mode_arg.value
    for kw in call_node.keywords:
        if kw.arg == "mode":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return ""


def get_func_name(node: ast.Call) -> str:
    """Get the function/method name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def check_call_usage(call_node: ast.Call) -> tuple[UsageType, str]:
    """
    Determine if a function call is a read or write operation.
    """
    method_name = get_func_name(call_node)

    # Special case for open()
    if method_name == "open":
        mode = get_open_mode(call_node)
        first_char = mode[0] if mode else "r"
        if first_char in {"w", "a", "x"}:
            return UsageType.WRITE, f"open() with mode '{mode}'"
        else:
            return UsageType.READ, f"open() with mode '{mode or 'r'}'"

    # Special case for h5py.File
    if method_name == "File":
        mode = get_open_mode(call_node)
        if mode and mode[0] in {"w", "a", "x"}:
            return UsageType.WRITE, f"h5py.File with mode '{mode}'"
        else:
            return UsageType.READ, f"h5py.File with mode '{mode or 'r'}'"

    # Check known write methods
    if method_name in WRITE_METHODS:
        return UsageType.WRITE, f"write method: {method_name}()"

    # Check known read methods
    if method_name in READ_METHODS:
        return UsageType.READ, f"read method: {method_name}()"

    # Heuristic based on method name patterns
    method_lower = method_name.lower()
    if any(w in method_lower for w in ["write", "save", "export", "dump"]):
        return UsageType.WRITE, f"method name pattern: {method_name}"
    if any(r in method_lower for r in ["read", "load", "import", "from_"]):
        return UsageType.READ, f"method name pattern: {method_name}"

    return UsageType.UNKNOWN, ""


# =============================================================================
# FUNCTION ANALYSIS FOR INTERPROCEDURAL TRACING
# =============================================================================


class FunctionAnalyzer(ast.NodeVisitor):
    """First pass: analyze function definitions to build parameter flow graph."""

    def __init__(self):
        self.functions: dict[str, FunctionInfo] = {}
        self._current_func: FunctionInfo | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        params = [arg.arg for arg in node.args.args]
        func_info = FunctionInfo(name=node.name, params=params)
        self._current_func = func_info

        for stmt in node.body:
            self.visit(stmt)

        self.functions[node.name] = func_info
        self._current_func = None

    def visit_Call(self, node: ast.Call):
        if self._current_func is None:
            self.generic_visit(node)
            return

        callee_name = get_func_name(node)
        if not callee_name:
            self.generic_visit(node)
            return

        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name) and arg.id in self._current_func.params:
                param_name = arg.id
                callee_param = f"arg{i}"
                if param_name not in self._current_func.param_flows:
                    self._current_func.param_flows[param_name] = []
                self._current_func.param_flows[param_name].append((callee_name, callee_param))

        for kw in node.keywords:
            if isinstance(kw.value, ast.Name) and kw.value.id in self._current_func.params:
                param_name = kw.value.id
                callee_param = kw.arg or "unknown"
                if param_name not in self._current_func.param_flows:
                    self._current_func.param_flows[param_name] = []
                self._current_func.param_flows[param_name].append((callee_name, callee_param))

        self.generic_visit(node)


def trace_param_usage(
    func_name: str,
    param_name: str,
    func_registry: dict[str, FunctionInfo],
    visited: set[str] | None = None,
    depth: int = 0,
) -> tuple[UsageType, str]:
    """
    Recursively trace parameter usage through function calls.
    """
    if visited is None:
        visited = set()

    key = f"{func_name}:{param_name}"
    if key in visited or depth > 10:
        return UsageType.UNKNOWN, ""
    visited.add(key)

    if func_name not in func_registry:
        return UsageType.UNKNOWN, ""

    func_info = func_registry[func_name]
    if param_name not in func_info.param_flows:
        return UsageType.UNKNOWN, ""

    for callee_name, callee_param in func_info.param_flows[param_name]:
        if callee_name in WRITE_METHODS:
            return UsageType.WRITE, f"{func_name}() -> {callee_name}()"

        if callee_name in READ_METHODS:
            return UsageType.READ, f"{func_name}() -> {callee_name}()"

        callee_lower = callee_name.lower()
        if any(w in callee_lower for w in ["write", "save", "export", "dump"]):
            return UsageType.WRITE, f"{func_name}() -> {callee_name}()"
        if any(r in callee_lower for r in ["read", "load", "import", "from_"]):
            return UsageType.READ, f"{func_name}() -> {callee_name}()"

        actual_param = callee_param
        if callee_name in func_registry and callee_param.startswith("arg"):
            try:
                idx = int(callee_param[3:])
                callee_func_info = func_registry[callee_name]
                if idx < len(callee_func_info.params):
                    actual_param = callee_func_info.params[idx]
            except (ValueError, IndexError):
                pass

        result, trace = trace_param_usage(
            callee_name, actual_param, func_registry, visited, depth + 1
        )
        if result != UsageType.UNKNOWN:
            return result, f"{func_name}() -> {trace}"

    return UsageType.UNKNOWN, ""


# =============================================================================
# MISC FILE VISITOR
# =============================================================================


class MiscFileVisitor(ast.NodeVisitor):
    """Find misc/ file references and determine their usage."""

    def __init__(
        self,
        source_lines: list[str],
        func_registry: dict[str, FunctionInfo],
        string_constants: dict[str, str] | None = None,
    ):
        self.source_lines = source_lines
        self.func_registry = func_registry
        self.direct_refs: list[tuple[str, int, str, UsageType, str]] = []
        # Each entry: (var_name, filename, lineno, context)
        self.var_assignments: list[tuple[str, str, int, str]] = []
        self.var_usages: dict[str, list[tuple[UsageType, str, int]]] = {}
        self.string_constants: dict[str, str] = string_constants or {}

    def get_line_context(self, lineno: int) -> str:
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    self.string_constants[var_name] = node.value.value

                misc_filename = is_misc_path(node.value, self.string_constants)
                if misc_filename:
                    context = self.get_line_context(node.lineno)
                    self.var_assignments.append((var_name, misc_filename, node.lineno, context))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self._check_direct_args(node)
        self._check_var_args(node)
        self.generic_visit(node)

    def _check_direct_args(self, node: ast.Call) -> None:
        """Check call arguments for direct string references."""
        all_args = list(node.args) + [kw.value for kw in node.keywords]

        for arg in all_args:
            if isinstance(arg, ast.Name):
                continue

            misc_filename = is_misc_path(arg, self.string_constants)
            if misc_filename:
                usage_type, reason = check_call_usage(node)
                context = self.get_line_context(node.lineno)
                self.direct_refs.append((misc_filename, node.lineno, context, usage_type, reason))

    def _check_var_args(self, node: ast.Call) -> None:
        """Check call arguments for usages of tracked variables."""
        func_name = get_func_name(node)

        # Check positional args
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name):
                self._record_var_usage(node, arg.id, func_name, arg_index=i)

        # Check keyword args
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                self._record_var_usage(node, kw.value.id, func_name, kw_arg=kw.arg)

    def _record_var_usage(
        self,
        node: ast.Call,
        var_name: str,
        func_name: str,
        arg_index: int | None = None,
        kw_arg: str | None = None,
    ) -> None:
        """Determine usage type and record it."""
        usage_type, reason = check_call_usage(node)

        # Interprocedural trace if unknown
        if usage_type == UsageType.UNKNOWN and func_name in self.func_registry:
            param_name = ""
            if arg_index is not None:
                func_info = self.func_registry[func_name]
                if arg_index < len(func_info.params):
                    param_name = func_info.params[arg_index]
            elif kw_arg is not None:
                param_name = kw_arg

            if param_name:
                usage_type, reason = trace_param_usage(func_name, param_name, self.func_registry)
                if usage_type != UsageType.UNKNOWN:
                    reason = f"via {reason}"

        if usage_type != UsageType.UNKNOWN:
            if var_name not in self.var_usages:
                self.var_usages[var_name] = []
            self.var_usages[var_name].append((usage_type, reason, node.lineno))


# =============================================================================
# NOTEBOOK ANALYSIS
# =============================================================================


def collect_string_constants(cells: list[NotebookCell]) -> dict[str, str]:
    """Collect simple string constant assignments from all cells."""
    constants: dict[str, str] = {}

    for cell in cells:
        if cell.tree is None:
            continue
        for node in ast.walk(cell.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            constants[target.id] = node.value.value

    return constants


def check_raw_line_usage(line: str) -> tuple[UsageType, str]:
    """Best-effort usage classification for non-AST fallback lines."""
    prefix = line.split("misc/", 1)[0]

    if re.search(r"\bopen\s*\(", prefix):
        mode_match = re.search(
            r"\bopen\s*\([^)]*misc/[^)]*,\s*mode\s*=\s*['\"]([^'\"]+)['\"]", line
        ) or re.search(r"\bopen\s*\([^)]*misc/[^)]*,\s*['\"]([^'\"]+)['\"]", line)
        mode = mode_match.group(1) if mode_match else "r"
        if mode and mode[0] in {"w", "a", "x"}:
            return UsageType.WRITE, f"raw fallback: open() with mode '{mode}'"
        return UsageType.READ, f"raw fallback: open() with mode '{mode}'"

    for method_name in WRITE_METHODS:
        if re.search(rf"(\.|\b){re.escape(method_name)}\s*\(", prefix):
            return UsageType.WRITE, f"raw fallback: write method {method_name}()"

    for method_name in READ_METHODS:
        if re.search(rf"(\.|\b){re.escape(method_name)}\s*\(", prefix):
            return UsageType.READ, f"raw fallback: read method {method_name}()"

    return UsageType.UNKNOWN, "raw fallback: usage not determined"


def find_raw_cell_references(notebook_name: str, cell: NotebookCell) -> list[FileReference]:
    """Find misc file references in a cell that could not be parsed as Python."""
    refs: list[FileReference] = []
    seen: set[str] = set()

    for lineno, line in enumerate(cell.source_lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        for match in RAW_MISC_PATH_PATTERN.finditer(line):
            filename = match.group("filename")
            if filename in seen or filename.startswith(".") or filename.endswith("/"):
                continue
            seen.add(filename)

            usage_type, reason = check_raw_line_usage(line)
            refs.append(
                FileReference(
                    notebook=notebook_name,
                    filename=filename,
                    line_number=lineno,
                    context=stripped,
                    usage_type=usage_type,
                    detection_reason=reason,
                )
            )

    return refs


def resolve_references(
    notebook_name: str,
    assignments: list[Assignment],
    var_usages: list[VariableUsage],
    direct_refs: list[DirectReference],
) -> list[FileReference]:
    """Resolve variable assignments to their first usage to determine file usage type."""
    # Index usages for efficient "first use after assignment" lookup.
    usages_by_var: dict[str, list[VariableUsage]] = {}
    for usage in var_usages:
        usages_by_var.setdefault(usage.var_name, []).append(usage)
    for var_name in usages_by_var:
        usages_by_var[var_name].sort(key=lambda x: (x.cell_index, x.line_number))

    candidates: list[tuple[tuple[int, int], FileReference]] = []

    def add_candidate(
        order: tuple[int, int],
        filename: str,
        lineno: int,
        context: str,
        usage_type: UsageType,
        reason: str,
        var_name: str = "",
    ) -> None:
        candidates.append(
            (
                order,
                FileReference(
                    notebook=notebook_name,
                    filename=filename,
                    line_number=lineno,
                    context=context,
                    usage_type=usage_type,
                    detection_reason=reason,
                    var_name=var_name,
                ),
            )
        )

    # Add direct references first (in notebook order).
    for ref in sorted(direct_refs, key=lambda x: (x.cell_index, x.line_number)):
        add_candidate(
            (ref.cell_index, ref.line_number),
            ref.filename,
            ref.line_number,
            ref.context,
            ref.usage_type,
            ref.reason,
            var_name="",
        )

    # Match each assignment to its first known usage after the assignment.
    for assign in sorted(assignments, key=lambda x: (x.cell_index, x.line_number)):
        first_use: VariableUsage | None = None
        for usage in usages_by_var.get(assign.var_name, []):
            if usage.cell_index > assign.cell_index or (
                usage.cell_index == assign.cell_index and usage.line_number > assign.line_number
            ):
                first_use = usage
                break

        if first_use is None:
            add_candidate(
                (assign.cell_index, assign.line_number),
                assign.filename,
                assign.line_number,
                assign.context,
                UsageType.UNKNOWN,
                f"variable '{assign.var_name}' usage not found",
                var_name=assign.var_name,
            )
            continue

        add_candidate(
            (assign.cell_index, assign.line_number),
            assign.filename,
            assign.line_number,
            assign.context,
            first_use.usage_type,
            f"variable '{assign.var_name}' used in {first_use.reason}",
            var_name=assign.var_name,
        )

    candidates.sort(key=lambda item: item[0])
    return prefer_import_references([ref for _, ref in candidates])


def prefer_import_references(references: list[FileReference]) -> list[FileReference]:
    """
    Deduplicate references by filename while preserving import-sensitive usage.

    A file that is ever read must stay classified as READ, even if the same
    notebook also writes that path later. WRITE-only files can safely be omitted
    from import_file_mapping.json.
    """
    priority = {
        UsageType.READ: 0,
        UsageType.UNKNOWN: 1,
        UsageType.WRITE: 2,
    }
    selected: dict[str, FileReference] = {}

    for ref in references:
        existing = selected.get(ref.filename)
        if existing is None or priority[ref.usage_type] < priority[existing.usage_type]:
            selected[ref.filename] = ref

    return list(selected.values())


def analyze_notebook(notebook_path: Path) -> list[FileReference]:
    """Analyze a single notebook with interprocedural analysis."""
    cells = extract_notebook_cells(notebook_path)

    # Build function registry (interprocedural tracing) from all valid cell ASTs.
    func_analyzer = FunctionAnalyzer()
    for cell in cells:
        if cell.tree is None:
            continue
        func_analyzer.visit(cell.tree)
    func_registry = func_analyzer.functions

    # Collect global string constants to help resolve concatenated paths.
    string_constants = collect_string_constants(cells)

    # Single pass over cells: collect direct refs, misc-path assignments, and var usages.
    assignments: list[Assignment] = []
    direct_refs: list[DirectReference] = []
    var_usages: list[VariableUsage] = []
    raw_refs: list[FileReference] = []

    for cell in cells:
        if cell.tree is None:
            raw_refs.extend(find_raw_cell_references(notebook_path.name, cell))
            continue
        visitor = MiscFileVisitor(cell.source_lines, func_registry, string_constants)
        visitor.visit(cell.tree)

        for var_name, filename, lineno, context in visitor.var_assignments:
            assignments.append(Assignment(var_name, filename, cell.index, lineno, context))

        for filename, lineno, context, usage_type, reason in visitor.direct_refs:
            direct_refs.append(
                DirectReference(filename, cell.index, lineno, context, usage_type, reason)
            )

        for var_name, uses in visitor.var_usages.items():
            for usage_type, reason, lineno in uses:
                var_usages.append(VariableUsage(var_name, cell.index, lineno, usage_type, reason))

    resolved_refs = resolve_references(notebook_path.name, assignments, var_usages, direct_refs)
    return prefer_import_references(resolved_refs + raw_refs)


def analyze_all_notebooks(root_dir: Path) -> AnalysisResult:
    """Analyze all notebooks in the project."""
    all_refs: list[FileReference] = []

    for notebook_path in sorted(root_dir.glob("*.ipynb")):
        all_refs.extend(analyze_notebook(notebook_path))

    return AnalysisResult(references=all_refs)


# =============================================================================
# MAPPING CHECK FUNCTIONS
# =============================================================================


def load_import_file_mapping(misc_dir: Path) -> dict[str, list[str]]:
    """Load import_file_mapping.json."""
    mapping_file = misc_dir / "import_file_mapping.json"
    if not mapping_file.exists():
        print(f"Error: Cannot find {mapping_file}")
        sys.exit(1)
    with open(mapping_file, encoding="utf-8") as f:
        return json.load(f)


def get_misc_files(misc_dir: Path) -> set[str]:
    """Get all file names in the misc directory (including subdirectories)."""
    files = set()
    for f in misc_dir.rglob("*"):
        if f.is_file():
            # Get relative path from misc_dir
            rel_path = f.relative_to(misc_dir)
            files.add(str(rel_path))
    return files


def check_notebook(
    notebook_path: Path,
    misc_files: set[str],
    import_mapping: dict[str, list[str]],
    errors: list[str],
    missing_mappings: dict[str, list[str]],
    actual_read_files: dict[str, set[str]],
) -> None:
    """Check a single notebook file using AST analysis."""
    notebook_name = notebook_path.name
    references = analyze_notebook(notebook_path)

    # Track actual READ files for this notebook
    read_files: set[str] = set()
    declared_files = set(import_mapping.get(notebook_name, []))

    for ref in references:
        # Skip WRITE operations - these are output files, not imports
        if ref.usage_type == UsageType.WRITE:
            continue

        filename = ref.filename

        # Skip files in the optional whitelist
        if filename in OPTIONAL_FILES:
            continue

        # Skip files referenced through optional variable name patterns
        # (e.g., history_fname for files that try to load previous run results)
        if ref.var_name in OPTIONAL_VAR_PATTERNS:
            continue

        # Track this as a READ file
        read_files.add(filename)

        # Check if the file exists in the misc directory
        if filename not in misc_files:
            errors.append(
                f"[FILE NOT FOUND] {notebook_name}: references 'misc/{filename}', "
                f"but the file does not exist in misc directory"
            )
            continue

        # Check if the file is declared in import_file_mapping.json
        if filename not in declared_files:
            errors.append(
                f"[NOT IN MAPPING] {notebook_name}: references 'misc/{filename}', "
                f"but it is not declared in import_file_mapping.json"
            )
            if notebook_name not in missing_mappings:
                missing_mappings[notebook_name] = []
            if filename not in missing_mappings[notebook_name]:
                missing_mappings[notebook_name].append(filename)

    # Store the actual read files for this notebook
    if read_files:
        actual_read_files[notebook_name] = read_files


def check_mapping_keys(
    import_mapping: dict[str, list[str]],
    notebooks: set[str],
    errors: list[str],
    invalid_keys: list[str],
) -> None:
    """Check if all keys in import_file_mapping.json are existing notebook files."""
    for notebook_name in import_mapping:
        if notebook_name not in notebooks:
            errors.append(
                f"[INVALID MAPPING KEY] '{notebook_name}' in import_file_mapping.json "
                f"does not exist as a notebook file"
            )
            invalid_keys.append(notebook_name)


def check_extra_mappings(
    import_mapping: dict[str, list[str]],
    actual_read_files: dict[str, set[str]],
    errors: list[str],
    extra_mappings: dict[str, list[str]],
) -> None:
    """Check for files in import_mapping that are not actually used."""
    for notebook_name, declared_files in import_mapping.items():
        actual_files = actual_read_files.get(notebook_name, set())

        for filename in declared_files:
            if filename not in actual_files:
                errors.append(
                    f"[EXTRA MAPPING] {notebook_name}: 'misc/{filename}' is declared "
                    f"in import_file_mapping.json but not actually used (READ)"
                )
                if notebook_name not in extra_mappings:
                    extra_mappings[notebook_name] = []
                extra_mappings[notebook_name].append(filename)


def update_import_file_mapping(
    misc_dir: Path,
    import_mapping: dict[str, list[str]],
    missing_mappings: dict[str, list[str]],
    extra_mappings: dict[str, list[str]] | None = None,
    invalid_keys: list[str] | None = None,
) -> None:
    """Update import_file_mapping.json: add missing, remove extra, delete invalid keys."""
    changes_made = False

    # Add missing mappings
    for notebook_name, files in missing_mappings.items():
        if notebook_name in import_mapping:
            existing = set(import_mapping[notebook_name])
            for f in files:
                if f not in existing:
                    import_mapping[notebook_name].append(f)
        else:
            import_mapping[notebook_name] = files
        changes_made = True

    # Remove extra mappings
    if extra_mappings:
        for notebook_name, files in extra_mappings.items():
            if notebook_name in import_mapping:
                import_mapping[notebook_name] = [
                    f for f in import_mapping[notebook_name] if f not in files
                ]
                # Remove empty entries
                if not import_mapping[notebook_name]:
                    del import_mapping[notebook_name]
                changes_made = True

    # Remove invalid keys (notebooks that don't exist)
    if invalid_keys:
        for notebook_name in invalid_keys:
            if notebook_name in import_mapping:
                del import_mapping[notebook_name]
                changes_made = True

    if not changes_made:
        return

    # Sort the mapping by notebook name for consistent output
    sorted_mapping = dict(sorted(import_mapping.items()))

    mapping_file = misc_dir / "import_file_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(sorted_mapping, f, indent=4)
        f.write("\n")

    print(f"\nUpdated {mapping_file}")
    if missing_mappings:
        print(f"  + Added mappings for {len(missing_mappings)} notebook(s)")
    if extra_mappings:
        total_removed = sum(len(files) for files in extra_mappings.values())
        print(
            f"  - Removed {total_removed} extra mapping(s) from {len(extra_mappings)} notebook(s)"
        )
    if invalid_keys:
        print(f"  - Removed {len(invalid_keys)} invalid notebook key(s)")


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================


def print_analysis_results(result: AnalysisResult, show_summary: bool = False) -> None:
    """Print analysis results in a readable format."""
    print("=" * 70)
    print("MISC FILE USAGE ANALYSIS (AST + Interprocedural)")
    print("=" * 70)

    ref_map: dict[tuple[str, str], FileReference] = {
        (ref.notebook, ref.filename): ref for ref in result.references
    }

    def print_section(
        title: str,
        icon: str,
        files_by_notebook: dict[str, list[str]],
    ) -> None:
        print(f"\n{icon} {title}:")
        print("-" * 50)
        if files_by_notebook:
            for notebook, files in sorted(files_by_notebook.items()):
                print(f"\n  📓 {notebook}:")
                for f in files:
                    ref = ref_map.get((notebook, f))
                    if ref:
                        print(f"     └─ {f}")
                        print(f"        Reason: {ref.detection_reason}")
                        print(f"        Context: {ref.context[:80]}...")
        else:
            print("  (none)")

    if not show_summary:
        print_section("WRITE OPERATIONS (first usage writes to file)", "📝", result.write_files)
        print()
        print_section("READ OPERATIONS (first usage reads from file)", "📖", result.read_files)
        print()
        print_section(
            "UNKNOWN OPERATIONS (cannot determine usage type)", "❓", result.unknown_files
        )

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_write = sum(len(files) for files in result.write_files.values())
    total_read = sum(len(files) for files in result.read_files.values())
    total_unknown = sum(len(files) for files in result.unknown_files.values())
    total = total_write + total_read + total_unknown

    print(f"\n  Total references analyzed: {total}")
    if total > 0:
        print(f"  ├─ WRITE operations: {total_write} ({100 * total_write / total:.1f}%)")
        print(f"  ├─ READ operations:  {total_read} ({100 * total_read / total:.1f}%)")
        print(f"  └─ UNKNOWN:          {total_unknown} ({100 * total_unknown / total:.1f}%)")
    else:
        print("  ├─ WRITE operations: 0")
        print("  ├─ READ operations:  0")
        print("  └─ UNKNOWN:          0")

    print(f"\n  Notebooks with WRITE: {len(result.write_files)}")
    print(f"  Notebooks with READ:  {len(result.read_files)}")
    print(f"  Notebooks with UNKNOWN: {len(result.unknown_files)}")


def output_json(result: AnalysisResult) -> None:
    """Output analysis results as JSON."""
    output = {
        "write_files": result.write_files,
        "read_files": result.read_files,
        "unknown_files": result.unknown_files,
        "details": [
            {
                "notebook": ref.notebook,
                "filename": ref.filename,
                "usage_type": ref.usage_type.value,
                "detection_reason": ref.detection_reason,
                "context": ref.context,
                "var_name": ref.var_name,
            }
            for ref in result.references
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check misc directory references in notebooks using AST analysis."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix missing mappings by adding them to import_file_mapping.json",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed usage analysis (WRITE/READ classification)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis results as JSON (requires --analyze)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary only (requires --analyze)",
    )
    args = parser.parse_args()
    if (args.json or args.summary) and not args.analyze:
        parser.error("--json and --summary require --analyze")
    return args


def run_analyze_mode(args: argparse.Namespace, root_dir: Path) -> None:
    """Run in analyze mode: show detailed usage analysis."""
    print(f"Analyzing notebooks in: {root_dir}", file=sys.stderr)
    result = analyze_all_notebooks(root_dir)

    if args.json:
        output_json(result)
    else:
        print_analysis_results(result, show_summary=args.summary)


def run_check_mode(args: argparse.Namespace, root_dir: Path, misc_dir: Path) -> None:
    """Run in check mode: validate mapping files."""
    print(f"Project root: {root_dir}")
    print(f"Misc directory: {misc_dir}")
    print("-" * 60)

    import_mapping = load_import_file_mapping(misc_dir)
    misc_files = get_misc_files(misc_dir)

    print(f"Found {len(misc_files)} files in misc directory")
    print(f"Found {len(import_mapping)} notebook mappings in import_file_mapping.json")
    print("-" * 60)

    notebooks = list(root_dir.glob("*.ipynb"))
    notebook_names = {nb.name for nb in notebooks}
    print(f"Found {len(notebooks)} notebook files")
    print("-" * 60)

    # Collect issues
    errors: list[str] = []
    missing_mappings: dict[str, list[str]] = {}
    extra_mappings: dict[str, list[str]] = {}
    invalid_keys: list[str] = []
    actual_read_files: dict[str, set[str]] = {}

    check_mapping_keys(import_mapping, notebook_names, errors, invalid_keys)

    for notebook_path in sorted(notebooks):
        check_notebook(
            notebook_path, misc_files, import_mapping, errors, missing_mappings, actual_read_files
        )

    check_extra_mappings(import_mapping, actual_read_files, errors, extra_mappings)

    # Report and optionally fix
    if not errors:
        print("\n[OK] All checks passed, no issues found!\n")
        sys.exit(0)

    print(f"\nFound {len(errors)} issues:\n")
    for error in errors:
        print(f"  X {error}")
    print()

    if not args.fix:
        sys.exit(1)

    # Apply fixes
    if missing_mappings or extra_mappings or invalid_keys:
        update_import_file_mapping(
            misc_dir, import_mapping, missing_mappings, extra_mappings, invalid_keys
        )

    # Check for unfixable errors
    file_not_found_errors = [e for e in errors if "[FILE NOT FOUND]" in e]
    if file_not_found_errors:
        print(
            f"\n[WARNING] {len(file_not_found_errors)} FILE NOT FOUND error(s) cannot be auto-fixed.\n"
        )
        sys.exit(1)

    print("\n[OK] All fixable issues have been fixed!\n")
    sys.exit(0)


def main():
    args = parse_args()
    root_dir = get_project_root()
    misc_dir = root_dir / "misc"

    if args.analyze:
        run_analyze_mode(args, root_dir)
    else:
        run_check_mode(args, root_dir, misc_dir)


if __name__ == "__main__":
    main()
