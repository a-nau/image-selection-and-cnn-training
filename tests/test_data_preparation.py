import sys
from pathlib import Path
import logging
import unittest
import shutil

ROOT = Path(__file__).parent.parent
sys.path.append(ROOT.as_posix())

from src.tools.keep_only_homogenous_backgrounds import (
    select_images_with_homogenous_boundaries,
)
from src.tools.preselect_images import create_parcel2d_distractors
from src.tools.generate_split import (
    generate_split_for_scraped_data,
    generate_split_for_scraped_data_according_to_json,
)
from src.maskrcnn.config import SCRAPED_DATA_PATH

logger = logging.getLogger(__name__)


class TestDataPreparation(unittest.TestCase):
    def test_select_homogenous_backgrounds(self):
        logger.info("Testing selection of images with homogeneous backgrounds")
        select_images_with_homogenous_boundaries(
            SCRAPED_DATA_PATH / "01_raw", SCRAPED_DATA_PATH / "02_homogeneous_bg"
        )

    def test_preselect_images(self):
        logger.info("Testing pre-selection of images")
        input_path_distractors = SCRAPED_DATA_PATH / "03_removed_bg"
        output_path_distractors = SCRAPED_DATA_PATH / "04_pre_selection"
        create_parcel2d_distractors(input_path_distractors, output_path_distractors)

    def test_generate_split_with_json(self):
        logger.info("Testing split generation")
        target = SCRAPED_DATA_PATH / "05_selection"
        source = SCRAPED_DATA_PATH / "04_pre_selection" / "selected"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        generate_split_for_scraped_data(source, remove_duplicates=True)
        # Scraped images
        generate_split_for_scraped_data_according_to_json(
            target, source / "splits.json",
        )
