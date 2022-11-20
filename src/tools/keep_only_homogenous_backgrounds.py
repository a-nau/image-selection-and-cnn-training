import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

import cv2
import tqdm

from src.selection.check_image import check_if_image_background_is_homogeneous
from src.maskrcnn.config import SCRAPED_DATA_PATH


def select_images_with_homogenous_boundaries(
    input_path: Path, target_path: Path, margin=0.02
):
    target_path.mkdir(exist_ok=True)
    for img_path in tqdm.tqdm(
        [f for f in input_path.glob("*") if f.suffix in [".jpg", ".png"]]
    ):
        img = cv2.imread(img_path.as_posix())
        if check_if_image_background_is_homogeneous(img, margin=margin, threshold=60):
            shutil.copy(img_path, target_path / img_path.name)


if __name__ == "__main__":
    select_images_with_homogenous_boundaries(
        SCRAPED_DATA_PATH / "01_raw", SCRAPED_DATA_PATH / "02_homogeneous_bg"
    )
