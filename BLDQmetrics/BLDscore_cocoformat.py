import matplotlib.pyplot as plt
import torch
import os
import pickle
from mmdet.datasets import build_dataset
from mmdet.datasets.api_wrappers import COCOeval
import numpy as np
import math
from astropy.modeling import models, fitting
import json
import pathlib
import re


# data_root_path = r"D:/datasets/cityscapes_coco/"
# val_split_path = r"D:/datasets/cityscapes_used/leftImg8bit/val/images/"
# gt_json_path = os.path.join(data_root_path, "instancesonly_filtered_gtFine_val.json")
# root_path = r"D:\code\mmdetection-master\mmdetection-master\Cityscapesexperiments"
data_root_path = r"D:/datasets/coco2017/annotations/"
gt_json_path = os.path.join(data_root_path, "instances_val2017.json")
root_path = r"D:\code\mmdetection-master\mmdetection-master\COCOexperiments"
# atssrh_result_path = os.path.join(root_path, "atss_r50_fpn_1x_5conv_cityscapes.pkl")
# atssrh_result_woNMS_path = os.path.join(root_path, "atss_r50_fpn_1x_5conv_cityscapes_noNMS.pkl")


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def read_predictions(path):
    extension = pathlib.Path(path).suffix
    if re.match(r".json", extension, re.IGNORECASE):
        return readjson(path)
    elif re.match(r".pkl", extension, re.IGNORECASE):
        return readpkl(path)


def readjson(path):
    with open(path, 'rb') as f:
        data = json.load(f)
        return data


def readpkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


def readgt(path):
    # dataset_type = 'CocoDataset'
    # test_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(
    #         type='MultiScaleFlipAug',
    #         img_scale=(2048, 1024),
    #         flip=False,
    #         transforms=[
    #             dict(type='Resize', keep_ratio=True),
    #             dict(type='RandomFlip'),
    #             dict(
    #                 type='Normalize',
    #                 mean=[123.675, 116.28, 103.53],
    #                 std=[58.395, 57.12, 57.375],
    #                 to_rgb=True),
    #             dict(type='Pad', size_divisor=32),
    #             dict(type='ImageToTensor', keys=['img']),
    #             dict(type='Collect', keys=['img'])
    #         ])
    # ]
    # test = dict(
    #     type=dataset_type,
    #     classes=('person', 'car', 'truck', 'rider', 'bicycle', 'motorcycle',
    #              'bus', 'train'),
    #     ann_file=path,
    #     img_prefix=val_split_path,
    #     pipeline=test_pipeline,
    #     test_mode=True)
    dataset_type = 'CocoDataset'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    test = dict(
        type=dataset_type,
        ann_file=r'D:/datasets/coco2017/annotations/instances_val2017.json',
        img_prefix=r'D:/datasets/coco2017/val2017/',
        pipeline=test_pipeline,
        test_mode=True
    )
    COCOdataset = build_dataset(test)
    return COCOdataset.coco.imgToAnns, COCOdataset.coco.imgs, COCOdataset


def sup_norm(sup_list, state):
    if state == "sigmoid":
        return 1. / (1. + np.exp(-sup_list))
    elif state == "MaxMin":
        return 1. / sup_list


def match_gt_pd_json(best_pd, pd_woNMS, gt, imgIdx, COCOdataset):
    """match the ground truth to predict bounding box

    Args:
        best_pd (_type_): the predict bbox by nms
        pd_woNMS (_type_): all bboxes
        gt (_type_): ground truth

    Returns:
        _type_: _description_
    """
    # 0. prepare data
    # for-image_number
    best_pd_count = 0
    best_count = 0
    sup_count = 0
    results = []
    # metric = COCOdataset.evaluate(best_pd)
    coco_det = COCOdataset.coco.loadRes(best_pd)
    cocoEval = COCOeval(COCOdataset.coco, coco_det, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    p = cocoEval.params
    catIds = p.catIds if p.useCats else [-1]
    for imgId in p.imgIds:
        for catId in catIds:
            gts = None
            dts = None
            if p.useCats:
                gts = cocoEval._gts[imgId, catId]
                dts = cocoEval._dts[imgId, catId]
            else:
                gts = [_ for cId in p.catIds for _ in cocoEval._gts[imgId, cId]]
                dts = [_ for cId in p.catIds for _ in cocoEval._dts[imgId, cId]]
            g = [g['bbox'] for g in gts]
            d = np.array([d['bbox'] for d in dts])
            if len(g) == 0 or len(d) == 0:
                continue

            for gt_item in g:
                gt_bbox_mid = np.array([gt_item[0], gt_item[1],
                                        gt_item[0] + gt_item[2], gt_item[1] + gt_item[3]])
                best_pd_list = np.array([d[:, 0], d[:, 1],
                                        d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]]).transpose()
                item_gt = np.array([gt_bbox_mid])
                # get the best_bboxes and the sup_bboxes according to class
                item_best_pd_list = np.array(best_pd_list[..., 0:4])
                item_sup_pd_list = np.array(best_pd_list[..., 0:4])
                item_sup_pd_socre_list = np.array(best_pd_list[..., -1])
                sup_count = sup_count + len(item_sup_pd_socre_list)

                # if no best_pd, find it in the set of all predictions (wo NMS)
                # if no pd_woNMS_list, continue
                if len(item_sup_pd_list) == 0:
                    continue
                elif len(item_best_pd_list) == 0:
                    # calculate the IoU of the selected gt and the predictions to match them
                    IoU_sup = get_IoU(item_sup_pd_list, item_gt)
                    # to the IoU_best, find the top_1
                    IoU_sup_torch = torch.tensor(IoU_sup)
                    supbest_value, supbest_indices = IoU_sup_torch.topk(1)
                    # to the IoU_sup, find all the IoU>0.6 ones (IoU between the best_pd and the sup_pd)
                    IoU_sup_torch_keep = IoU_sup_torch > 0.5
                    if supbest_value < 0.5:
                        item = {"image_id": imgId, "gt_bboxes": item_gt,
                                "best_pd_bbox": np.array([]),
                                "best_pd_bbox_IoU": np.array([0.0]),
                                "best_bbox": np.array([]),
                                "best_bbox_IoU": np.array([0.0]),
                                "sup_bboxes_06": np.array([]),
                                "sup_bboxes_06_IoU": np.array([0.0])}
                    else:
                        item = {"image_id": imgId, "gt_bboxes": item_gt,
                                "best_pd_bbox": np.array([]),
                                "best_pd_bbox_IoU": np.array([0.0]),
                                "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                                "best_bbox_IoU": np.array(supbest_value.numpy()),
                                "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                                "sup_bboxes_06_IoU": np.array(IoU_sup_torch[IoU_sup_torch_keep.numpy()].numpy())}
                        best_count += 1
                else:
                    # calculate the IoU of the selected gt and the predictions to match them
                    IoU_best = get_IoU(item_best_pd_list, item_gt)
                    IoU_sup = get_IoU(item_sup_pd_list, item_gt)
                    # to the IoU_best, find the top_1
                    IoU_best_torch = torch.tensor(IoU_best)
                    IoU_sup_torch = torch.tensor(IoU_sup)
                    best_value, best_indices = IoU_best_torch.topk(1)
                    supbest_value, supbest_indices = IoU_sup_torch.topk(1)
                    # to the IoU_sup, find all the IoU>0.6 ones (IoU between the best_pd and the sup_pd)
                    IoU_sup_torch_keep = IoU_sup_torch > 0.5
                    if best_value < 0.5 and supbest_value < 0.5:
                        item = {"image_id": imgId, "gt_bboxes": item_gt,
                                "best_pd_bbox": np.array([]),
                                "best_pd_bbox_IoU": np.array([0.0]),
                                "best_bbox": np.array([]),
                                "best_bbox_IoU": np.array([0.0]),
                                "sup_bboxes_06": np.array([]),
                                "sup_bboxes_06_IoU": np.array([0.0])}
                    elif best_value < 0.5 and supbest_value >= 0.5:
                        item = {"image_id": imgId, "gt_bboxes": item_gt,
                                "best_pd_bbox": np.array([]),
                                "best_pd_bbox_IoU": np.array([0.0]),
                                "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                                "best_bbox_IoU": np.array(supbest_value.numpy()),
                                "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                                "sup_bboxes_06_IoU": np.array(IoU_sup_torch[IoU_sup_torch_keep.numpy()].numpy())}
                        best_count += 1
                    else:
                        # get coordinates
                        item = {"image_id": imgId, "gt_bboxes": item_gt,
                                "best_pd_bbox": np.array([item_best_pd_list[best_indices]]),
                                "best_pd_bbox_IoU": np.array(best_value.numpy()),
                                "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                                "best_bbox_IoU": np.array(supbest_value.numpy()),
                                "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                                "sup_bboxes_06_IoU": np.array(IoU_sup_torch[IoU_sup_torch_keep.numpy()].numpy())}
                        best_count += 1
                        best_pd_count += 1
                results.append(item)


    return results, best_pd_count, best_count, sup_count


def get_IoU(pred, target, eps=1e-7):
    n, _ = pred.shape
    target = target.repeat(n, axis=0)
    # overlap
    lt = np.maximum(pred[:, :2], target[:, :2])
    rb = np.minimum(pred[:, 2:], target[:, 2:])
    wh = np.maximum(0, rb - lt)
    overlap = wh[:, 0] * wh[:, 1]
    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union
    return ious


def data_trans(match_results, mode=["cls_score", "IoU"], state="MaxMin"):
    best_pd_x_list = np.array([])
    best_pd_y_list = np.array([])
    best_x_list = np.array([])
    best_y_list = np.array([])
    sup_x_list = np.array([])
    sup_y_list = np.array([])
    repeat_times = 1

    mode_list = ["IoU", "1/ΔIoU", "x1", "x2", "y1", "y2", "cls_score", "edges", "best_edges", "worst_edges"]
    index_list = []
    for key in mode_list:
        if key == mode[0]:
            index_list.append("x")
        elif key == mode[1]:
            index_list.append("y")
        else:
            index_list.append("")

    if "edges" in mode:
        repeat_times = 4

    # judge mode and calculate distance
    for item in match_results:
        if len(item["sup_bboxes_06"]) < 1:
            continue
        if "x1" in mode or "x2" in mode or "y1" in mode or "y2" in mode or \
                "edges" in mode or "best_edges" in mode or "worst_edges" in mode:
            if len(item["best_pd_bbox"]) == 0:
                continue
        if "IoU" in mode:
            key_idx = mode_list.index("IoU")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate((best_pd_x_list, item["best_pd_bbox_IoU"]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox_IoU"]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06_IoU"]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate((best_pd_y_list, item["best_pd_bbox_IoU"]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox_IoU"]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06_IoU"]))
        if "1/ΔIoU" in mode:
            key_idx = mode_list.index("1/ΔIoU")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate(
                        (best_pd_x_list, sup_norm(1.0 - item["best_pd_bbox_IoU"], "MaxMin")))
                    best_x_list = np.concatenate((best_x_list, sup_norm(1.0 - item["best_bbox_IoU"], "MaxMin")))
                    sup_x_list = np.concatenate((sup_x_list, sup_norm(1.0 - item["sup_bboxes_06_IoU"], "MaxMin")))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate(
                        (best_pd_y_list, sup_norm(1.0 - item["best_pd_bbox_IoU"], "MaxMin")))
                    best_y_list = np.concatenate((best_y_list, sup_norm(1.0 - item["best_bbox_IoU"], "MaxMin")))
                    sup_y_list = np.concatenate((sup_y_list, sup_norm(1.0 - item["sup_bboxes_06_IoU"], "MaxMin")))
        if "x1" in mode:
            key_idx = mode_list.index("x1")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate(
                        (best_pd_x_list, item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate(
                        (best_pd_y_list, item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0]))
        if "x2" in mode:
            key_idx = mode_list.index("x2")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate(
                        (best_pd_x_list, item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate(
                        (best_pd_y_list, item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2]))
        if "y1" in mode:
            key_idx = mode_list.index("y1")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate(
                        (best_pd_x_list, item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate(
                        (best_pd_y_list, item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1]))
        if "y2" in mode:
            key_idx = mode_list.index("y2")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate(
                        (best_pd_x_list, item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate(
                        (best_pd_y_list, item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]))
        if "cls_score" in mode:
            key_idx = mode_list.index("cls_score")
            if index_list[key_idx] == "x":
                for repeat_time in range(repeat_times):
                    best_pd_x_list = np.concatenate((best_pd_x_list, item["best_pd_bbox_scores"]))
                    best_x_list = np.concatenate((best_x_list, item["best_bbox_scores"]))
                    sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06_scores"][0]))
            else:
                for repeat_time in range(repeat_times):
                    best_pd_y_list = np.concatenate((best_pd_y_list, item["best_pd_bbox_scores"]))
                    best_y_list = np.concatenate((best_y_list, item["best_bbox_scores"]))
                    sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06_scores"][0]))
        if "edges" in mode:
            key_idx = mode_list.index("edges")
            if index_list[key_idx] == "x":
                best_pd_x_list = np.concatenate(
                    (best_pd_x_list, item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0]))
                best_pd_x_list = np.concatenate(
                    (best_pd_x_list, item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2]))
                best_pd_x_list = np.concatenate(
                    (best_pd_x_list, item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1]))
                best_pd_x_list = np.concatenate(
                    (best_pd_x_list, item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_x_list = np.concatenate((best_x_list, item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                sup_x_list = np.concatenate((sup_x_list, item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]))

            else:
                best_pd_y_list = np.concatenate(
                    (best_pd_y_list, item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0]))
                sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0]))
                best_pd_y_list = np.concatenate(
                    (best_pd_y_list, item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2]))
                sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2]))
                best_pd_y_list = np.concatenate(
                    (best_pd_y_list, item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1]))
                sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1]))
                best_pd_y_list = np.concatenate(
                    (best_pd_y_list, item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_y_list = np.concatenate((best_y_list, item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                sup_y_list = np.concatenate((sup_y_list, item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]))
        if "best_edges" in mode:
            key_idx = mode_list.index("best_edges")
        if "worst_edges" in mode:
            key_idx = mode_list.index("worst_edges")
            if index_list[key_idx] == "x":
                best_pd_x_value = np.concatenate((item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0],
                                                  item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2],
                                                  item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1],
                                                  item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_pd_x_value_abs = np.abs(best_pd_x_value)
                best_pd_x_value_idx = np.argmax(best_pd_x_value_abs)
                best_pd_x_value = np.array([best_pd_x_value[best_pd_x_value_idx]])
                best_pd_x_list = np.concatenate((best_pd_x_list, best_pd_x_value))
                best_x_value = np.concatenate((item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0],
                                               item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2],
                                               item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1],
                                               item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_x_value_abs = np.abs(best_x_value)
                best_x_value_idx = np.argmax(best_x_value_abs)
                best_x_value = np.array([best_x_value[best_x_value_idx]])
                best_x_list = np.concatenate((best_x_list, best_x_value))
                sup_x_value = np.stack((item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0],
                                        item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2],
                                        item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1],
                                        item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]), axis=1)
                sup_x_value_abs = np.abs(sup_x_value)
                sup_x_value_idx = np.argmax(sup_x_value_abs, axis=1)
                for sup_idx in range(len(sup_x_value_idx)):
                    sup_x_item = np.array([sup_x_value[sup_idx, sup_x_value_idx[sup_idx]]])
                    sup_x_list = np.concatenate((sup_x_list, sup_x_item))
            else:
                best_pd_y_value = np.concatenate((item["best_pd_bbox"][..., 0] - item["gt_bboxes"][0, 0],
                                                  item["best_pd_bbox"][..., 2] - item["gt_bboxes"][0, 2],
                                                  item["best_pd_bbox"][..., 1] - item["gt_bboxes"][0, 1],
                                                  item["best_pd_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_pd_y_value_abs = np.abs(best_pd_y_value)
                best_pd_y_value_idx = np.argmax(best_pd_y_value_abs)
                best_pd_y_value = np.array([best_pd_y_value[best_pd_y_value_idx]])
                best_pd_y_list = np.concatenate((best_pd_y_list, best_pd_y_value))
                best_y_value = np.concatenate((item["best_bbox"][..., 0] - item["gt_bboxes"][0, 0],
                                               item["best_bbox"][..., 2] - item["gt_bboxes"][0, 2],
                                               item["best_bbox"][..., 1] - item["gt_bboxes"][0, 1],
                                               item["best_bbox"][..., 3] - item["gt_bboxes"][0, 3]))
                best_y_value_abs = np.abs(best_y_value)
                best_y_value_idx = np.argmax(best_y_value_abs)
                best_y_value = np.array([best_y_value[best_y_value_idx]])
                best_y_list = np.concatenate((best_y_list, best_y_value))
                sup_y_value = np.stack((item["sup_bboxes_06"][..., 0] - item["gt_bboxes"][0, 0],
                                        item["sup_bboxes_06"][..., 2] - item["gt_bboxes"][0, 2],
                                        item["sup_bboxes_06"][..., 1] - item["gt_bboxes"][0, 1],
                                        item["sup_bboxes_06"][..., 3] - item["gt_bboxes"][0, 3]), axis=1)
                sup_y_value_abs = np.abs(sup_y_value)
                sup_y_value_idx = np.argmax(sup_y_value_abs, axis=1)
                for sup_idx in range(len(sup_y_value_idx)):
                    sup_y_item = np.array([sup_y_value[sup_idx, sup_y_value_idx[sup_idx]]])
                    sup_y_list = np.concatenate((sup_y_list, sup_y_item))
    return best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list


def astropy_gaussian(x, y, mean=0., std=10.0):
    g_init = models.Gaussian1D(amplitude=1., mean=mean, stddev=std)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y, maxiter=300)
    return g.mean.value, g.stddev.value


def fit_gaussian_2(best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list):
    best_pd_mean, best_pd_std = astropy_gaussian(best_pd_x_list, best_pd_y_list, mean=0.0, std=50.0)
    best_mean, best_std = astropy_gaussian(best_x_list, best_y_list, mean=0.0, std=50.0)
    sup_mean, sup_std = astropy_gaussian(sup_x_list, sup_y_list, mean=0.0, std=50.0)
    return best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std


def draw_gaussian(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std,
                  mode=["cls_score", "IoU"], path=''):
    color = ["blue", "orange", "gray", "red", "yellow", "purple"]
    # g_mean + 2* g_std
    min_values = []
    max_values = []

    min_values.append(best_pd_mean - 3 * best_pd_std)
    max_values.append(best_pd_mean + 3 * best_pd_std)
    min_values.append(best_mean - 3 * best_std)
    max_values.append(best_mean + 3 * best_std)
    min_values.append(sup_mean - 3 * sup_std)
    max_values.append(sup_mean + 3 * sup_std)
    min_value = min(min_values)
    max_value = max(max_values)

    plt.figure("gaussian")
    x = np.linspace(min_value, max_value, 20 * int(max_value - min_value) + 1)

    # plt.title("best_pd")
    y = normal_distribution(x, best_pd_mean, best_pd_std)
    plt.plot(x, y, color[0], label="Predictions w/ NMS")
    # plt.axvline(x=best_pd_mean, color="black", linestyle='--')
    y = normal_distribution(x, best_mean, best_std)
    plt.plot(x, y, color[1], label="Best predictions")
    # plt.axvline(x=best_mean, color="black", linestyle='--')
    # plt.text(g_mean + 0.5, 0.1, "({},{})".format(np.round(g_mean, 2), np.round(g_std, 2)), fontsize=12)
    y = normal_distribution(x, sup_mean, sup_std)
    plt.plot(x, y, color[2], label="Predictions w/o NMS")
    # plt.axvline(x=sup_mean, color="black", linestyle='--')
    plt.axvline(x=0, color="green", label="Ground-truth")

    plt.xlabel(mode[0], fontsize=16)
    plt.ylabel(mode[1], fontsize=16)
    plt.legend(fontsize=12)
    # plt.show()
    plt.savefig(path)
    plt.clf()
    return


def main():
    # best_pd = readpkl(with_nms_path)
    # woNMS_pd = readpkl(without_nms_path)
    file = open('Cityscapesexperiments/test_borderdet_coco.txt', 'w')
    # file.write('best_pd_mean', 'best_pd_std', 'best_mean', 'best_std', 'sup_mean', 'sup_std', 'label')
    file.write('best_pd_mean best_pd_std best_mean best_std sup_mean sup_std label\n')

    gtAnns, imgIdx, COCOdataset = readgt(gt_json_path)
    best_pd = read_predictions(
        r"D:\code\mmdetection-master\mmdetection-master\COCOexperiments\BorderDet\coco800\json_out\coco_instances_results.json"
    )
    # woNMS_pd = read_predictions(without_nms_path[i])
    # label = os.path.basename(labels[i])
    label = r"BorderDet_src_coco"

    match_results, best_pd_count, best_count, sup_count = match_gt_pd_json(best_pd, best_pd, gtAnns, imgIdx,
                                                                           COCOdataset)
    best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list = \
        data_trans(match_results, mode=["worst_edges", "IoU"], state="MaxMin")

    best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std = \
        fit_gaussian_2(best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list)
    print(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std, label, file=file)
    print((abs(best_pd_mean) + abs(best_pd_std)) / best_pd_count * 1000,
          (abs(best_mean) + abs(best_std)) / best_count * 1000,
          (abs(sup_mean) + abs(sup_std)) / sup_count * 1000000, file=file)
    # print(best_pd_count, best_count, sup_count, file=file)
    print(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std, label)
    print(best_pd_count, best_count, sup_count)
    print((abs(best_pd_mean) + abs(best_pd_std)) / best_pd_count * 1000,
          (abs(best_mean) + abs(best_std)) / best_count * 1000,
          (abs(sup_mean) + abs(sup_std)) / sup_count * 1000000)

    return


if __name__ == '__main__':
    main()
