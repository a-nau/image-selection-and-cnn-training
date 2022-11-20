import json
import random
from pathlib import Path

import tqdm
from pycocotools._mask import frPyObjects, merge
from pycocotools.mask import area
from itertools import combinations


def get_area_from_coco_segmentation_list(segmentation, image_size):
    if type(segmentation) == list:
        h, w = image_size
        rles = frPyObjects(segmentation, h, w)
        rle = merge(rles)
    else:
        rle = segmentation
    segmentation_area = area(rle)
    return segmentation_area


def get_bounding_box(points2d, coco_formatted=True):
    import cv2
    import numpy as np

    points2d = np.asarray(points2d).reshape(-1, 2).astype(np.float32)
    if len(points2d) > 0:
        bbox2d = list(cv2.boundingRect(points2d))
        if not coco_formatted:
            bbox2d = np.array(
                [[bbox2d[0], bbox2d[1]], [bbox2d[0] + bbox2d[2], bbox2d[1] + bbox2d[3]]]
            ).reshape(2, 2)
    else:
        bbox2d = []
    return bbox2d


def load_data(rgb_path: Path, image_id: int, annotation_id: int):
    annotation_file = rgb_path.with_suffix(".json")
    if not annotation_file.exists():
        return None, "annotation file"
    with open(annotation_file.as_posix(), "r") as f:
        annotations = json.load(f)
    images = annotations["images"]
    if len(images) != 1:
        return None, "too many images"
    image = images[0]
    image["id"] = image_id
    annotations = annotations["annotations"]
    for i, annotation in enumerate(annotations):
        annotation["image_id"] = image_id
        annotation["area"] = int(
            get_area_from_coco_segmentation_list(
                annotation["segmentation"], (image["height"], image["width"])
            )
        )
        annotation["id"] = annotation_id + i
    return image, annotations


def create_parcel2d_json(
    dataset_path: Path,
    split: str,
    image_id: int,
    annotation_id: int,
    num_samples: int = None,
    dataset_name="parcel2d",
    blending_methods=None,
) -> (int, int):
    """
    For info on json format see
    - https://cocodataset.org/#format-data
    """
    data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "box", "id": 0, "name": "normal box"},
            {"supercategory": "box", "id": 1, "name": "damaged box"},
        ],
    }
    samples = list((dataset_path / split).rglob("*.jpg"))
    samples = [s for s in samples if s.name[6:-6] in blending_methods]
    if num_samples is not None:
        samples = random.sample(samples, min(num_samples, len(samples)))
        split = f"{split}_{str(num_samples)}"
    count = 0
    for i, sample in enumerate(tqdm.tqdm(samples), start=image_id + 1):
        image, annotations = load_data(sample, i, annotation_id)
        if image is not None:
            data["images"].append(image)
            data["annotations"].extend(annotations)
            annotation_id += len(annotations)
            count += 1
        else:
            pass
    dataset_name = (
        f"{dataset_name}_{'_'.join(blending_methods)}"
        if len(blending_methods) != 4
        else dataset_name
    )

    with open((dataset_path / f"{dataset_name}_{split}.json").as_posix(), "w") as f:
        json.dump(data, f)
    print(
        f"Saved {count}/{len(samples)} files for {split} to {dataset_path / f'{dataset_name}_{split}.json'}"
    )
    return image_id, annotation_id


def create_parcel2d_subset_jsons(
    dataset_path,
    image_id,
    annotation_id,
    number_of_samples,
    dataset_name=None,
    blending_methods=None,
):
    for split in ["train", "test", "validation"]:
        image_id, annotation_id = create_parcel2d_json(
            dataset_path,
            split,
            image_id,
            annotation_id,
            num_samples=number_of_samples,
            dataset_name=dataset_name,
            blending_methods=blending_methods,
        )
    return image_id, annotation_id


def create_parcel2d_subsets_blending(dataset_path, blending_methods=None):
    blending_methods = (
        ["gaussian", "motion", "none", "poisson-fast"]
        if blending_methods is None
        else blending_methods
    )

    blending_method_combinations = []
    for n in range(1, len(blending_methods) + 1):
        blending_method_combinations += list(combinations(blending_methods, n))

    for dataset_name in ["parcel2d_plain", "parcel2d_maskrcnn", "parcel2d_manual"]:
        for blending_methods in blending_method_combinations:
            for num_samples in [None, 10, 100, 1000]:
                image_id, annotation_id = create_parcel2d_subset_jsons(
                    dataset_path / dataset_name,
                    image_id=0,
                    annotation_id=0,
                    number_of_samples=num_samples,
                    dataset_name=dataset_name,
                    blending_methods=blending_methods,
                )
