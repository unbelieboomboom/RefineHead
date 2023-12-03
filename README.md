# BLDQ metrics & PDEM


## Introduction

The code is being reformatted for better readability!
A major update will come soon!

<img src="https://github.com/unbelieboomboom/RefineHead\configs\RefineHead\RefineHead_structure">



## Installation

PDEM is built on [MMetection](https://github.com/open-mmlab/mmdetection).

Please refer to [Installation](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md) for the basic usage of MMDetection.



## Starting with PDEM

Training PDEM (baseline=ATSS):

```shell
python tools/train.py \
    configs/PDEM/PDEM_atss_r50_fpn_1x_cityscapes.py 
```
PDEM + CoordConv:

```shell
python tools/train.py \
    configs/PDEM/PDEM_atss_coord_r50_fpn_1x_cityscapes.py 
```

PDEM + GFL:

```shell
python tools/train.py \
    configs/PDEM/PDEM_gfl_r50_fpn_1x_cityscapes.py 
```

PDEM + TOOD:

```shell
python tools/train.py \
    configs/PDEM/PDEM_tood_r50_fpn_1x_cityscapes.py 
```

## Obtaining BLDQ scores

1. First, save the detection results of your model into json files or into pkl files::

```shell

```

2. Then, evaluate the detection results through:

(Reformatted code will update soon!)

```shell
python BLDQmetrics/BLDQ_Cityscapes.py
```

## Model Zoo

(After the trained models are uploaded to the Cloud, the data will be updated quickly!)

| Model  | PDEM | mAP(%) | BLD | BBLD | IBLD | Configs | Download |
|--------|------|--------|-----|------|------|---------|----------|
| `ATSS` | x    |        |     |      |      |         |          |
| `ATSS` | o    |        |     |      |      |         |          |
| `TOOD` | x    |        |     |      |      |         |          |
| `TOOD` | o    |        |     |      |      |         |          |

('o' denotes using PDEM while 'x' denotes not.)