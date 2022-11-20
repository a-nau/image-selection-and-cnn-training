import shutil
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())
from src.maskrcnn.config import SCRAPED_DATA_PATH
from src.selection.utils import load_predictor, check_and_save_images
from src.selection.check_image import (
    check_image_with_maskrcnn,
    check_image_plain,
    check_image_manually,
)


def create_parcel2d_objects(input_folder: Path, target_base_folder: Path):
    check_image_functions = {
        "maskrcnn": partial(check_image_with_maskrcnn, predictor=load_predictor()),
        "plain": check_image_plain,
        "manual": check_image_manually,
    }
    for dataset_name, check_image_function in check_image_functions.items():
        print(f"{'#'*20} Starting with {dataset_name} {'#'*20}")
        target_folder = target_base_folder / f"parcel2d_{dataset_name}"
        if target_folder.exists():
            shutil.rmtree(target_folder)
        target_folder.mkdir()
        check_and_save_images(input_folder, target_folder, check_image_function)


def create_parcel2d_distractors(input_folder: Path, target_folder: Path):
    print(f"{'#' * 20} Starting with distractors {'#' * 20}")
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir()
    check_and_save_images(input_folder, target_folder, check_image_manually)


if __name__ == "__main__":

    input_path_distractors = SCRAPED_DATA_PATH / "03_removed_bg"
    output_path_distractors = SCRAPED_DATA_PATH / "04_pre_selection"
    create_parcel2d_distractors(input_path_distractors, output_path_distractors)
    # Here 3 different checks are applied
    # create_parcel2d_objects(input_path_objects, output_path_objects)
