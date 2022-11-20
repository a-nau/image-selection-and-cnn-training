import json
from collections import Counter
from pathlib import Path
from typing import Callable, Union, Dict

import cv2
import tqdm
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

ROOT = Path(__file__).parent.parent.parent


def load_predictor():
    checkpoint_path = ROOT / "data" / "pretrained" / "openimages_10k.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find pre-trained checkpoint at {str(checkpoint_path)}"
        )

    config_file = ROOT / "src" / "maskrcnn" / "configs" / "openimages.yaml"
    cfg = get_cfg()
    cfg.SOLVER.FREEZE_BACKBONE = False
    cfg.merge_from_file(config_file.as_posix())
    cfg.MODEL.WEIGHTS = checkpoint_path.as_posix()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95
    predictor = DefaultPredictor(cfg)
    return predictor


def find_images(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    image_paths = [
        f
        for f in path.rglob("*")
        if f.suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    ]
    return image_paths


def check_and_save_images(
    source: Path, target: Path, check_image: Callable, min_size_kb=100
):
    image_paths = find_images(source)
    print(f"Found {len(image_paths)} images")
    flags = []
    for image_path in tqdm.tqdm(image_paths):
        if (image_path.stat().st_size / 2 ** 10) < min_size_kb:
            flag = "too_small"
        else:
            image_org = cv2.imread(
                image_path.as_posix(), cv2.IMREAD_UNCHANGED
            )  # Load RGBA images from rembg
            flag = check_image(image_org)
        flags.append(flag)
        target_folder = target / flag
        target_folder.mkdir(exist_ok=True)
        # Save tight cropped image
        image = Image.open(image_path.as_posix())
        cropped = image.crop(image.getbbox())
        cropped.save(target_folder / image_path.name)
    c = Counter(flags)
    print(c)


def load_json(file: Union[str, Path]):
    with file.open("r") as fp:
        return json.load(fp)


def save_json(file: Path, data: Dict):
    with file.open("w") as f:
        json.dump(obj=data, fp=f, indent=4)
