import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Union

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.selection.utils import save_json
from src.maskrcnn.config import SCRAPED_DATA_PATH


def remove_common_path(path: Union[str, Path], reference: Union[str, Path]) -> Path:
    path = Path(path)
    reference = Path(reference)

    path = path.expanduser().absolute()
    reference = reference.expanduser().absolute()
    common_path = os.path.commonpath([str(path), str(reference)])
    return path.relative_to(Path(common_path))


def generate_split_for_scraped_data(
    image_dir, test_ratio=0.2, validation_ratio=0.2, remove_duplicates=True
):
    """
    Generate split for all images in a folder
    """
    image_files = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))

    # Remove duplicates
    if remove_duplicates:
        image_file_ids = [f.stem.split("_")[-1] for f in image_files]
        duplicate_ids = [
            item for item, count in Counter(image_file_ids).items() if count > 1
        ]
        image_files = [
            f for f in image_files if f.stem.split("_")[-1] not in duplicate_ids
        ] + [[f for f in image_files if id in f.name][0] for id in duplicate_ids]
        print(f"Found {len(image_files)} images in {image_dir}")

    image_names = [remove_common_path(f, image_dir).as_posix() for f in image_files]
    train_ratio = 1 - test_ratio - validation_ratio

    x_train, x_test, y_train, y_test = train_test_split(
        image_names, ["box"] * len(image_names), test_size=1 - train_ratio
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )
    data = {
        "path": image_dir.as_posix(),
        "test": x_test,
        "train": x_train,
        "validation": x_val,
    }
    save_json(image_dir / "splits.json", data)


def generate_split_for_scraped_data_according_to_json(
    image_dir: Path, global_split_json: Path
):
    """
    Generate split for special folders. We have a global split in `global_split_json`, so that all test sets are
    real test sets with no overlap -> it is possible to test and validate on the same sub-dataset.
    Since duplicates are removed in the global_split_json, we need to look for matching IDs not full file names.
    """
    image_files = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))

    with global_split_json.open("r") as f:
        global_split = json.load(f)
    global_image_file_ids = {
        key: [f.split("_")[-1] for f in global_split[key]]
        for key in ["test", "train", "validation"]
    }

    image_names = [remove_common_path(f, image_dir).as_posix() for f in image_files]
    data = {
        "path": image_dir.as_posix(),
        "test": [
            f for f in image_names if f.split("_")[-1] in global_image_file_ids["test"]
        ],
        "train": [
            f for f in image_names if f.split("_")[-1] in global_image_file_ids["train"]
        ],
        "validation": [
            f
            for f in image_names
            if f.split("_")[-1] in global_image_file_ids["validation"]
        ],
    }
    print(
        f"Found {len(image_files)} images; { {key: len(val) for key, val in data.items() if key != 'path'} }"
    )
    save_json(image_dir / "splits.json", data)


def generate_split_for_SUNRGBD(image_dir, validation_ratio=0.5):
    """
    Data is in 'test' and 'train' folder. Use all train data and split test in test and val
    """
    image_files = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))
    image_names = [remove_common_path(f, image_dir) for f in image_files]
    x_train = [f.as_posix() for f in image_names if f.parent.name == "train"]
    x_test_val = [f.as_posix() for f in image_names if f.parent.name == "test"]

    x_val, x_test, y_val, y_test = train_test_split(
        x_test_val, ["box"] * len(x_test_val), test_size=1 - validation_ratio
    )
    data = {
        "path": image_dir.as_posix(),
        "test": x_test,
        "train": x_train,
        "validation": x_val,
    }
    save_json(image_dir / "splits.json", data)


def generate_split_for_SUN397(
    image_dir, test_ratio=0.2, validation_ratio=0.2, min_size_kb=600
):
    with (image_dir / "ClassName.txt").open("r") as f:
        class_names = f.read().splitlines()
    ignore_categories = [
        "archive",
    ]
    class_names = [
        class_name
        for class_name in class_names
        if class_name.split("/")[2] not in ignore_categories
    ]
    train_ratio = 1 - test_ratio - validation_ratio

    x_train, x_test, y_train, y_test = train_test_split(
        class_names, ["box"] * len(class_names), test_size=1 - train_ratio
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )
    image_files = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))
    image_files = [f for f in image_files if (f.stat().st_size / 2 ** 10) > min_size_kb]

    def get_images_from_list(class_list):
        image_paths = [
            remove_common_path(f, image_dir).as_posix()
            for f in image_files
            if (f'/{"/".join(f.parts[-3:-1])}' in class_list)
        ]
        return image_paths

    data = {
        "path": image_dir.as_posix(),
        "test": get_images_from_list(x_test),
        "train": get_images_from_list(x_train),
        "validation": get_images_from_list(x_val),
    }
    save_json(image_dir / "splits.json", data)


if __name__ == "__main__":
    # Generate "global" split so the same models are always in the same split (necessary if you want to test different image selection strategies as in our paper)
    generate_split_for_scraped_data(
        SCRAPED_DATA_PATH / "04_pre_selection" / "selected", remove_duplicates=True
    )

    # Scraped images
    generate_split_for_scraped_data_according_to_json(
        SCRAPED_DATA_PATH / "05_selection",
        SCRAPED_DATA_PATH / "04_pre_selection" / "selected" / "splits.json",
    )

    # # Distractors
    # generate_split_for_scraped_data(distractor_dir)
    #
    # # Background
    # generate_split_for_SUNRGBD(background_folder)
    # generate_split_for_SUN397(background_folder)
