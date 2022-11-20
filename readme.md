[![arxiv](http://img.shields.io/badge/paper-arxiv.2210.09814-B31B1B.svg)][arxiv]
[![project page](https://img.shields.io/badge/website-project%20page-informational.svg)][project page]

# CNN Training and Image Selection

You can use this code for

- [CNN Training](#CNN-Training): Using detectron2 to train a CNN on your own dataset
- [Image Selection](#Pipeline-for-Image-Selection): Process and select images scraped from the internet to create your
  own dataset

If you are interested in generating your own instance segmentation dataset, we highly recommend to check
our [project page][project page] for more detailed information. Quick overview:

<p align="center">
    <img src="https://a-nau.github.io/parcel2d/static/images/overview.png" alt="Overview" height="350"/>
    <br>
    <span style="font-size: small; margin-left: 20px; margin-right: 20px">
      <b>Figure:</b> 
      Overview of our dataset generation pipeline: (1) We
      scrape images from popular image search engines. (2) We
      use and compare three different methods for image selection,
      i.e. basic pre-processing, manual selection and CNN-based
      selection. (3) The objects of interest and the distractors are
      pasted onto a randomly selected background. (4) We use
      four different blending methods to ensure invariance to local
      pasting artifacts as suggested by <a href="https://arxiv.org/abs/1708.01642">Dwibedi et al.</a>
    </span>
    <br>
</p>

## CNN Training

To run a training on our 5 [example images](data/parcel2d_demo/train) run:

```shell
python src/tools/train_maskrcnn.py --config-file ./src/maskrcnn/configs/maskrcnn.yaml --gpus "0" --num-gpus 1 --num-machines 1
```

- To add your own dataset, you need to register in the [register_datasets.py](src/maskrcnn/data/register_datasets.py).
- To check your results qualitatively you can
  use [detectron_qualitative_evaluation.ipynb](src/notebooks/detectron_qualitative_evaluation.ipynb)

To compute the final performance adjust and run [eval_maskrcnn.py](src/tools/eval_maskrcnn.py)

```shell
python src/tools/eval_maskrcnn.py
```

## Pipeline for Image Selection

To generate your own dataset from scraped images, please follow these steps. We added some minimal test data, to show
you the process.

- Paste the data your scraped (for details see [this](https://github.com/a-nau/easy-image-scraping))
  into `data/scraped/01_raw`
- (Optional) If you want to, apply the check for a homogeneous background by running
  ```shell
  python src/tools/keep_only_homogenous_backgrounds.py
  ```
- Apply background removal by running [`remove_background.sh`](scripts/remove_background.sh)
    - Note: Docker is necessary
    - If you want to use homogeneous backgrounds only, please adjust the paths
      in [`remove_background.sh`](scripts/remove_background.sh)
- Run pre-selection by filtering out small images and probably unsuccessful background removals. In addition,
  tight-cropping is applied
  ```shell
  python src/tools/preselect_images.py
  ```
- Perform final selection (e.g. manually) of images. Paste all relevant images into `data/scraped/05_selection`
- Generate training, validation and test split
  ```shell
  python src/tools/generate_split.py
  ```

You can now use this split for training!

To inspect your dataset including annotations
check [detectron_dataset_visualization.ipynb](src/notebooks/detectron_dataset_visualization.ipynb).

## Installation

### Locally

Everything should work fine, using Docker. If you want to develop locally please set up your environment using

```shell
pip install -r requirements.txt
```

Afterwards install `torch` and `detectron2`:

```shell
# For CPU (not recommended)
pip install torch==1.10.0 torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# For GPU with CUDA 11.3
pip install torch==1.10.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

To verify that everything works run

```shell
python -m unittest
```

### Docker

First build

```shell
source scripts/GPU/docker_build.sh      # GPU
source scripts/docker_build.sh          # CPU
```

and then run

```shell
source scripts/GPU/docker_run.sh        # GPU
source scripts/docker_run.sh            # CPU
```

And inside the container run the test with

```shell
python -m unittest
```

## Citation

If you use this code for scientific research, please consider citing

```latex
@inproceedings{naumannScrapeCutPasteLearn2022,
  title = {Scrape, Cut, Paste and Learn: Automated Dataset Generation Applied to Parcel Logistics},
  booktitle = {{{IEEE Conference}} on {{Machine Learning}} and Applications} ({{ICMLA}})},
  author = {Naumann, Alexander and Hertlein, Felix and Zhou, Benchun and Dörr, Laura and Furmans, Kai},
  date = {2022},
  note = {to appear in},
}
```

- Paper: [arxiv][arxiv]
- If you are interested in generating your own instance segmentation dataset, we highly recommend to check
  our [project page][project page]

## Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

[arxiv]: https://arxiv.org/abs/2210.09814

[project page]: https://a-nau.github.io/parcel2d
