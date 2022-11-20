#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

from detectron2.data import (
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model

logger = logging.getLogger(__name__)


def freeze_backbone(model):
    for child_name, child_module in model.named_children():
        if child_name == "backbone":
            for param in child_module.parameters():
                param.requires_grad = False


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        self._best_ap = -1
        self._store_checkpoint = False
        super().__init__(cfg)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        if freeze_at <= 5:
            pass
        elif freeze_at == 6:
            cfg.defrost()
            cfg.MODEL.BACKBONE.FREEZE_AT = 2
            cfg.freeze()
        else:
            raise NotImplementedError("Illegal freeze stage")

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        if freeze_at == 6:
            freeze_backbone(model)
            logger.info("Backbone weights frozen")
        return model

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge([640, 800], max_size=1200),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(angle=[0, 15]),
            T.RandomContrast(intensity_min=0.9, intensity_max=1.1),
            T.RandomBrightness(intensity_min=0.9, intensity_max=1.1),
            T.RandomSaturation(intensity_min=0.9, intensity_max=1.1),
            T.RandomLighting(scale=255),
        ]

        return build_detection_train_loader(
            cfg, mapper=DatasetMapper(cfg, True, augmentations=augs)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        return COCOEvaluator(
            dataset_name, tasks=("bbox", "segm",), output_dir=output_dir,
        )
