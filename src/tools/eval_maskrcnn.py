import os
import sys
from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

from src.maskrcnn.data import register_datasets  # noqa
from src.maskrcnn.engine.train_loop import Trainer


def load_cfg():
    config_file = ROOT / "src" / "maskrcnn" / "configs" / "maskrcnn.yaml"
    checkpoint = ROOT / "data" / "pretrained" / "openimages_10k.pth"

    cfg = get_cfg()
    cfg.merge_from_file(config_file.as_posix())
    cfg.MODEL.WEIGHTS = checkpoint.as_posix()
    cfg.freeze()
    default_setup(cfg, None)
    return cfg


def eval(cfg, dataset_name):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    dataset_loader = trainer.build_test_loader(cfg, dataset_name)
    evaluator = COCOEvaluator(dataset_name, tasks=("bbox", "segm"),)
    res = inference_on_dataset(trainer.model, dataset_loader, evaluator)
    print(f"Results: {res}")


if __name__ == "__main__":
    cfg = load_cfg()
    eval(cfg, dataset_name="parcel2d_demo_train")
