from functools import partial
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from src.maskrcnn.config import OPENIMAGES_PATH, PARCEL2D_PATH

meta = [
    {"name": "normal box", "color": [255, 255, 25], "id": 0},
]


def register_dataset(dataset_name: str, json_file: Path, image_root: Path):
    DatasetCatalog.register(
        dataset_name, partial(load_coco_json, json_file, image_root)
    )
    # Set meta data
    things_ids = [k["id"] for k in meta]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    thing_classes = [k["name"] for k in meta]
    thing_colors = [k["color"] for k in meta]
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=str(json_file.resolve()), image_root=image_root.as_posix(), **metadata
    )


dataset_paths = {
    "openimages": OPENIMAGES_PATH,
    "parcel2d_demo": PARCEL2D_PATH,
}

for dataset_name, dataset_path in dataset_paths.items():
    for json_file in dataset_path.glob("*.json"):
        dataset_split_name = f"{dataset_name}_{json_file.stem}"
        print(f"Registering: {dataset_split_name}")
        register_dataset(
            dataset_name=dataset_split_name,
            json_file=json_file,
            image_root=dataset_path,
        )
print("Registered all datasets")
