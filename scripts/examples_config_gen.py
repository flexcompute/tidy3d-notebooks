import copy
import glob
import json
import os
import random
import re
from pathlib import Path

import emoji
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

excludePathList = [
    "SimpleModeSolverGUI.ipynb",
    "WaveguideBendSimulator.ipynb",  # SCEM-5165: JupyterHub does not support tk.Tk()
    "CreatingGeometryUsingTrimesh.ipynb",  # JupyterHub does not support trimesh show
]
excludeCatalogList = [
    "Inverse Design with the Adjoint Plugin (Deprecated)",  # SCEM-9024: hide old Inverse Design examples
]


def get_data_file_mapping():
    with open(PROJECT_ROOT / "misc" / "import_file_mapping.json") as f:
        return json.load(f)


class Complie:
    enter_file = ""
    toc = {}
    rst_files = []
    example_type_map = {}

    def __init__(self, enter_file) -> None:
        self.enter_file = enter_file
        self.data_files_mapping = get_data_file_mapping()

    def convert_custom_emoji_to_code(self, text):
        # covert Sphinx emoji codes into shortcodes (from '|:fire:|' to ':fire:')
        text = re.sub(r"\|:(.*?):\|", r":\1:", text)

        # convert emoji shortcodes into unicode characters
        text = emoji.emojize(text, language="alias")

        return text

    def start(self):
        self.files_to_compile()
        self.get_h1_submenu()
        pass

    def files_to_compile(self):
        directory = os.path.dirname(self.enter_file)
        self.rst_files = glob.glob(os.path.join(directory, "**", "*.rst"), recursive=True)

    def get_h1_submenu(self):
        self.toc = self.pase_rst_file(self.enter_file)
        self.excluding_none_ipynb_files(self.toc)
        self.write_toc()

    def get_rst_file_path(self, base_directory, path):
        _path = path
        if not _path.endswith(".rst"):
            _path += ".rst"
        directory = os.path.dirname(base_directory)
        if not os.path.exists(directory):
            raise ValueError(f"{directory} is not exist.")
        return os.path.join(directory, _path)

    def is_jupyter_file(self, line):
        if "../" not in line:
            return False
        _path = line
        if not _path.endswith(".ipynb"):
            _path += ".ipynb"
        filename = os.path.basename(_path)
        full_path = PROJECT_ROOT / filename
        return full_path.exists()

    def is_rst_directory(self, file_path, line):
        if "../" in line:
            return False
        rst_file = self.get_rst_file_path(file_path, line)
        return os.path.exists(rst_file)

    def is_path(self, file_path, line, toc):
        line = line.strip()
        if self.is_rst_directory(file_path, line):
            path = self.get_rst_file_path(file_path, line)
            sub_toc = self.pase_rst_file(path, toc)
            if "title" in toc:
                sub_toc["category"] = toc["title"]
            if "enable_more_examples" in toc:
                sub_toc["enable_more_examples"] = toc["enable_more_examples"]
            if "type" in toc:
                sub_toc["type"] = toc["type"]
            toc["menus"].append(sub_toc)
            pass
        elif self.is_jupyter_file(line):
            path = os.path.join("./", os.path.basename(line))
            doc = {}
            if not path.strip().endswith(".ipynb"):
                doc["path"] = path.strip() + "/"
            else:
                _path = path.strip().replace(".ipynb", "")
                doc["path"] = _path + "/"
            doc["enable"] = True
            if "title" in toc:
                doc["category"] = toc["title"]
            if "enable_more_examples" in toc:
                doc["enable_more_examples"] = toc["enable_more_examples"]
            if "type" in toc:
                doc["type"] = toc["type"]
            toc["menus"].append(doc)

    def pase_rst_file(self, file_path, parent_toc=None):
        toc = {}
        previous_line = ""
        with open(file_path) as file:
            for line_num, line in enumerate(file):
                if len(line.strip()) > 0 and line.strip() == "=" * len(line.strip()):
                    toc["title"] = self.convert_custom_emoji_to_code(previous_line)
                    toc["menus"] = []
                    toc["enable"] = True
                elif len(line.strip()) > 0 and line.strip() == "-" * len(line.strip()):
                    h2Toc = {}
                    h2Toc["title"] = self.convert_custom_emoji_to_code(previous_line)
                    h2Toc["enable"] = True
                    h2Toc["menus"] = []
                    if "Case Studies" in h2Toc["title"]:
                        h2Toc["enable_library"] = True
                        h2Toc["type"] = "example-library"
                        h2Toc["enable_more_examples"] = True
                    else:
                        h2Toc["type"] = "python-tutorial"
                        h2Toc["enable_more_examples"] = False

                    # toc['menus'].append(h2Toc)
                    toc = h2Toc
                elif (
                    len(line.strip()) > 0
                    and line.strip() == "~" * len(line.strip())
                    and bool(parent_toc)
                ):
                    h3Toc = {}
                    h3Toc["title"] = self.convert_custom_emoji_to_code(previous_line)
                    h3Toc["enable"] = True
                    if "title" in parent_toc:
                        h3Toc["category"] = parent_toc["title"]
                    if "enable_more_examples" in parent_toc:
                        h3Toc["enable_more_examples"] = parent_toc["enable_more_examples"]
                    if "type" in parent_toc:
                        h3Toc["type"] = parent_toc["type"]

                    h3Toc["menus"] = []
                    toc = h3Toc
                else:
                    previous_line = line.strip()
                    if bool(toc):
                        self.is_path(file_path, line, toc)

        file.close()
        return toc

    def excluding_none_ipynb_files(self, json):
        if "menus" in json:
            for index, child in enumerate(json["menus"]):
                if "menus" in child and len(child["menus"]) == 0 and "path" not in child:
                    del json["menus"][index]
                else:
                    self.excluding_none_ipynb_files(child)

    def convert_to_yml(self, json):
        if "menus" in json:
            for child in json["menus"]:
                if "path" in child and "menus" not in child:
                    title = self.get_jupyter_title(child["path"])
                    thumbnail = self.get_jupyter_thumbnail(child["path"], title)
                    child["name"] = title
                    if thumbnail:
                        child["thumbnail"] = thumbnail
                elif "menus" in child:
                    self.convert_to_yml(child)

    def get_jupyter_title(self, path):
        headings = []
        file_path = PROJECT_ROOT / f"{path}"
        if file_path.suffix != ".ipynb":
            file_path = file_path.with_suffix(".ipynb")
        with open(file_path) as f:
            nb = json.load(f)
            for cell in nb["cells"]:
                cell_type = cell.get("cell_type", "")
                source = cell.get("source", [])
                if (
                    cell_type == "markdown"
                    and isinstance(source, list)
                    and source
                    and source[0].startswith("# ")
                ):
                    headings.append(source[0].strip("#").strip())

        return headings[0] if len(headings) > 0 else ""

    def get_jupyter_thumbnail(self, path, desc):
        file_path = PROJECT_ROOT / f"{path}"
        if file_path.suffix != ".ipynb":
            file_path = file_path.with_suffix(".ipynb")
        with open(file_path) as f:
            nb = json.load(f)
            metadata = nb["metadata"]
            feature_image = metadata.get("feature_image", "")
            image_name = os.path.basename(feature_image)
            relative_path = os.path.normpath(feature_image)
            feature_image_absolute_path = PROJECT_ROOT / relative_path
            if feature_image_absolute_path.is_file() and feature_image_absolute_path.exists():
                return {"image": f"/assets/tidy3d/examples/image/{image_name}", "description": desc}
        return None

    def get_category_map(self, toc=None, map=None):
        if toc is None:
            toc = []
        if map is None:
            map = {}
        for menu in toc:
            if "path" in menu and menu["enable"]:
                type = menu["type"]
                if map.get(type) is None:
                    map[type] = []
                map[type].append({"path": menu["path"], "name": menu["name"]})
            elif "menus" in menu:
                map = self.get_category_map(menu["menus"], map)
        return map

    def get_example_pagination(self, menu):
        type = menu["type"]
        example_list = self.example_type_map[type]
        if len(example_list) == 0:
            return {"previous": False, "next": False}
        menu_index = next(
            (i for i, item in enumerate(example_list) if item.get("path") == menu["path"]), -1
        )
        if menu_index == 0:
            return {"previous": False, "next": example_list[1]}
        elif menu_index > 0 and menu_index < len(example_list) - 1:
            return {"previous": example_list[menu_index - 1], "next": example_list[menu_index + 1]}
        else:
            return {"previous": example_list[menu_index - 1], "next": False}

    def insert_more_examples(self, toc=None, next_category_examples=None):
        if toc is None:
            toc = []
        if next_category_examples is None:
            next_category_examples = []
        for i, menu in enumerate(toc):
            if "path" in menu and menu["enable"]:
                menu["more_examples"] = self.random_examples(i, toc, next_category_examples)
                menu["pagination"] = self.get_example_pagination(menu)
            elif "menus" in menu:
                next_category_examples = self.get_other_examples(i, toc)
                self.insert_more_examples(menu["menus"], next_category_examples)

    def get_other_examples(self, index, examples):
        more_examples = []
        for i, example in enumerate(examples):
            if i != index and "menus" in example:
                temp_examples = [
                    menu
                    for menu in example["menus"]
                    if "path" in menu and "enable" in menu and menu["enable"]
                ]
                more_examples = more_examples + temp_examples
        return more_examples

    def get_example_by_index(self, index, examples):
        deep_copied_example = copy.deepcopy(examples[index])
        # remove unused fields
        if "more_examples" in deep_copied_example:
            del deep_copied_example["more_examples"]
        if "enable_more_examples" in deep_copied_example:
            del deep_copied_example["enable_more_examples"]
        if "enable" in deep_copied_example:
            del deep_copied_example["enable"]
        return deep_copied_example

    def omit(self, dict, keys_to_omit):
        return {key: dict[key] for key in dict if key not in keys_to_omit}

    def omit_more_example(self, exmaple):
        return self.omit(copy.deepcopy(exmaple), ["pagination", "category", "type"])

    def random_examples(self, index, category_examples, next_category_examples):
        category_examples_length = len(category_examples)
        loop_length = 3 if category_examples_length > 4 else category_examples_length - 1
        random_integers = []
        more_examples = []
        while len(random_integers) < loop_length:
            random_i = random.randint(0, category_examples_length - 1)
            random_example = category_examples[random_i]
            if (
                random_i != index
                and random_i not in random_integers
                and "enable" in random_example
                and random_example["enable"]
            ):
                random_integers.append(random_i)

        random_integers.sort()
        if len(random_integers) < 3 and len(next_category_examples) > 3 - len(random_integers):
            extra_random_integers = []
            while len(extra_random_integers) < 3 - len(random_integers):
                extra_random_i = random.randint(0, len(next_category_examples) - 1)
                if extra_random_i not in extra_random_integers:
                    extra_random_integers.append(extra_random_i)
            if len(extra_random_integers) > 0:
                random_integers.append(extra_random_integers)

        more_examples = []
        for i in random_integers:
            if isinstance(i, list):
                for index in i:
                    more_example = self.omit_more_example(
                        self.get_example_by_index(index, next_category_examples)
                    )
                    more_examples.append(more_example)
            else:
                more_example = self.omit_more_example(
                    self.get_example_by_index(i, category_examples)
                )
                more_examples.append(more_example)
        return more_examples

    def write_toc(self):
        def list_menus(result, menus):
            for menu in menus:
                if menu.get("path"):
                    path = menu["path"]
                    category = menu["category"]
                    without_suffix_path = path[0 : len(path) - 1]
                    path = without_suffix_path + ".ipynb"
                    filename = os.path.basename(path)
                    if (filename in excludePathList) or (category in excludeCatalogList):
                        continue
                    filename_without_ext = os.path.basename(without_suffix_path)
                    data = {
                        "title": menu["name"],
                        "source": filename,
                        "catalog": category,
                        "type": "notebook",
                    }
                    if filename in self.data_files_mapping:
                        data["dataFiles"] = self.data_files_mapping[filename]
                    elif filename_without_ext in self.data_files_mapping:
                        data["dataFiles"] = self.data_files_mapping[filename_without_ext]
                    result.append(data)
                elif menu.get("menus"):
                    list_menus(result, menu.get("menus"))
                else:
                    continue

        self.convert_to_yml(self.toc)
        result = []
        list_menus(result, self.toc["menus"])
        toc_yml = yaml.dump(result)

        with open(CONFIG_FILE, "w") as f:
            f.write(toc_yml)


def main():
    compile = Complie(PROJECT_ROOT / "docs" / "index.rst")
    compile.start()


if __name__ == "__main__":
    main()
