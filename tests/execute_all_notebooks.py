import os
import pathlib
import pytest
from testbook import testbook

# # note: these libraries throw Deprecation warnings in python 3.9, so they are ignored in pytest.ini
# import nbformat
# from nbconvert.preprocessors import CellExecutionError
# from nbconvert.preprocessors import ExecutePreprocessor

# sys.path.append("tidy3d")

# ep = ExecutePreprocessor(timeout=3000, kernel_name="python3")

# Get the directory where the current script (or notebook) is located
NOTEBOOKS_DIR = pathlib.Path(__file__).parent.parent.resolve()

# List all .ipynb files in the directory, excluding the .ipynb_checkpoints directory
notebook_filenames_all = [
    NOTEBOOKS_DIR / f  # Use '/' for path concatenation, which is supported by pathlib
    for f in os.listdir(NOTEBOOKS_DIR)  # Directly use NOTEBOOK_DIR, which is already a Path object
    if (f.endswith(".ipynb")) and (not f.startswith(".ipynb_checkpoints"))
]

# sort alphabetically
notebook_filenames_all.sort()


# Loop through notebooks in notebook_filenames and verify they all run.
@pytest.mark.parametrize("notebook_path", argvalues=notebook_filenames_all)
def test_notebooks(notebook_path):
    @testbook(notebook_path, execute=True)
    def test_notebook(tb):
        # Verifies that it executes without error
        return 0

    print("Running: " + str(notebook_path))
    test_notebook(notebook_path)
    return 0
