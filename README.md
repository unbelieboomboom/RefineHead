# BLDQ metrics & PDEM


## Introduction

The code is being reformatted for better readability!
A major update will come soon!

## Installation

PDEM is built on [MMetection](https://github.com/open-mmlab/mmdetection).

Please refer to [Installation](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md) for the basic usage of MMDetection.


## Motivation

IoU can only quantify the proportion of the overlap area between the prediction box and the groundtruth, 
failing to describe the gap between the non-overlapping part and the ground-truth. Even with identical IoU scores, 
the detected Bboxes may have significant gaps in identifying and locating instances

<img src="https://github.com/unbelieboomboom/RefineHead/images/deviation.png">

## Obtaining BLDQ scores

BLDQ metrics are used to evaluate the quality of the Bboxes with the same IoU but different geometric shapes.

(Reformatted code will update soon, sample detection results, .pkl and .json files, are uploading!)

1. First, save the detection results of your model into json files or into pkl files::

```shell

```

2. Then, evaluate the detection results through:



```shell
python BLDQmetrics/BLDQ_Cityscapes.py
```


## Architecture of PDEM

<img src="https://github.com/unbelieboomboom/RefineHead/images/PDEM.png">


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


## Model Zoo


| Model  | PDEM | dataset    | mAP(%) | BLD  | BBLD  | IBLD   | Configs                                                            | Download                                                              |
|--------|------|------------|--------|------|-------|--------|--------------------------------------------------------------------|-----------------------------------------------------------------------|
| `ATSS` | o    | Cityscapes | `40.4` | 10.2 | `7.9` | 20.8   | [cfg](https://github.com/unbelieboomboom/RefineHead/configs/PDEM/PDEM_atss_r50_fpn_1x_cityscapes.py) | [提取码：4pww](https://pan.baidu.com/s/1qVprp6nL7o_hGf15oDPEMA?pwd=4pww)  |
| `ATSS` | x    | Cityscapes | 38.2   | 11.2 | 8.7   | 24.6   |                                                                                                      |                                                                       |
| `ATSS` | o    | COCO       | 39.5   | 4.5  | `3.0` | 7.7    | [cfg](https://github.com/unbelieboomboom/RefineHead/configs/PDEM/PDEM_atss_r50_fpn_1x_coco.py)       | [提取码：lwlk](https://pan.baidu.com/s/1YHuFDxMzKaFCo6XfHURImw?pwd=lwlk)  |
| `ATSS` | x    | COCO       | 38.8   | 4.6  | 3.2   | 7.7    |                                                                                                      |                                                                       |
| `TOOD` | o    | Cityscapes | 40.2   | 8.9  | 8.6   | `20.5` | [cfg](https://github.com/unbelieboomboom/RefineHead/configs/PDEM/PDEM_tood_r50_fpn_1x_cityscapes.py) | [提取码：gber](https://pan.baidu.com/s/1OSwzXigVV29SJYW186ozqg?pwd=gber)  |
| `TOOD` | x    | Cityscapes | 39.3   | 10.2 | 8.7   | 22.3   |                                                                                                      |                                                                       |
| `TOOD` | o    | COCO       | `41.6` | 4.5  | 3.5   | `9.1`  | [cfg](https://github.com/unbelieboomboom/RefineHead/configs/PDEM/PDEM_tood_r50_fpn_1x_coco.py)       | [提取码：m86g](https://pan.baidu.com/s/1jzBOl1CrhDXpURBd2t416A?pwd=m86g)  |
| `TOOD` | x    | COCO       | 41.3   | 4.4  | 3.3   | 9.8    |                                                                                                      |                                                                       |

('o' denotes using PDEM while 'x' denotes not.)