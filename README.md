# RefineHead
RefineHead is a highly explainable and flexible FCN-based head network using predictive distribution information for bbox refinement.


## Introduction

RefineHead is a FCN-based head network for bbox refinement.

<img src="https://github.com/unbelieboomboom/RefineHead\configs\RefineHead\RefineHead_structure">

- **Flexible Architecture**

    RefineHead only attaches a few extra 3×3 convolution layers to a plain bbox regression head, 
so it can be flexibly integrated into FCN-based object detectors to boost performance.


- **Explainable**

    RefineHead fuses explicit predictive distribution with implicit localization features for further
bbox refinement, thus it has a closer predictive distribution to the ground-truth distribution.


## Installation

RefinedHead is built on [MMetection](https://github.com/open-mmlab/mmdetection).

Please refer to [Installation](docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection.



### Code of RefineHead

Training RefineHead:

```shell
python tools/train.py \
    configs/RefineHead/RefineHead_r50_fpn_1x_cityscapes.py 
```

### code of Distribution

1. Replace the vanilla [mmdet\models\base_dense_head.py] by our 
[[mmdet\models\base_dense_head_RH.py]](mmdet\models\base_dense_head_RH.py),
rename [base_dense_head_RH.py] to [base_dense_head.py],
and ensure you have [core\visualization\RefineHead_image.py](core\visualization\RefineHead_image.py)


2. Set the name, the path, and the ground-truth of one input image in [base_dense_head.py]
```shell
    # TODO: visualization
    png_name = "frankfurt_000000_012121_leftImg8bit.png"
    png_path = "/dataset/cityscapes/leftImg8bit/val/images/frankfurt_000000_012121_leftImg8bit.png"
    json_path = "/dataset/cityscapes/instancesonly_filtered_gtFine_val.json"
```

3. Save the bbox predictions into a .json file by running:
```shell
python tools/train.py \
    demo/visual_RefineHead.py 
```

4. Use the .json file and the predicted coordinates for fitting Gaussian distributions

```shell
python tools/train.py \
    demo/draw_distribution.py 
```

## Pretrained model

 Models pretrained on Cityscapes are already available [here](https://pan.baidu.com/s/1dpr62o15ShxUTT96yKdN1A?pwd=atud)
(extract code：atud)

## Model Zoo

Waiting for updates.