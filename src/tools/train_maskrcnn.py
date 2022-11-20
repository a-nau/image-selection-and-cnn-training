import os
import sys
from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch, default_argument_parser
from detectron2.utils.logger import setup_logger

ROOT = Path(__file__).parent.parent.parent
sys.path.append(ROOT.as_posix())

import src.maskrcnn.data.register_datasets  # noqa
from src.maskrcnn.engine.train_loop import Trainer


def load_config_from_file(config_file: Path, freeze=True):
    cfg = get_cfg()
    cfg.merge_from_file(config_file.as_posix())
    if freeze:
        cfg.freeze()
    default_setup(cfg, None)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskrcnn"
    )
    return cfg


def parse_args(args):
    parser = default_argument_parser()
    parser.add_argument("--gpus", default="0", help="Set GPUs that should be used")
    args = parser.parse_args(args)
    print("Command Line Args:", args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args


def main(args):
    config_file = (
        ROOT / "src" / "maskrcnn" / "configs" / "maskrcnn.yaml"
        if args.config_file == ""
        else Path(args.config_file)
    )
    print(f"Using config: {config_file}")
    cfg = load_config_from_file(config_file)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,  # default 1 in default_argument_parser
        dist_url=args.dist_url,
        args=(args,),
    )
