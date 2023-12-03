import matplotlib.pyplot as plt
import torch
import os
import pickle
from mmdet.datasets import build_dataset
import numpy as np
import math
from astropy.modeling import models, fitting
from dirtree import Dir_tree
import json
import pathlib
import re


def _get_all_file_path(directory):
    file_paths = []
    file_names = []
    file = ''
    # 使用os.walk遍历目录及其子目录
    for root, directories, files in os.walk(directory):
        for filename in files:
            if '.pkl' in filename:
                file = os.path.join(root, filename)
            # 获取文件的完整路径
            file_names.append(file_names)
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)

    return file_paths, file_names, file


data_root_path = r"D:/datasets/cityscapes_coco/"
val_split_path = r"D:/datasets/cityscapes_used/leftImg8bit/val/images/"
gt_json_path = os.path.join(data_root_path, "instancesonly_filtered_gtFine_val.json")
root_path = r"D:\code\mmdetection-master\mmdetection-master\Cityscapesexperiments"
# atssrh_result_path = os.path.join(root_path, "atss_r50_fpn_1x_5conv_cityscapes.pkl")
# atssrh_result_woNMS_path = os.path.join(root_path, "atss_r50_fpn_1x_5conv_cityscapes_noNMS.pkl")


dirtree = Dir_tree(root_path, 0)
dirtree.remove_node('pkl', 3)
# labels = dirtree.tree2dirlist(level=[2])
labels = []
files = []
for directory in dirtree.tree2dirlist(level=[3]):
    file_path, file_names, file = _get_all_file_path(directory=directory)
    files.append(file)

with_nms_path = []
without_nms_path = []
for i, item in enumerate(files):
    if i % 2 == 0:
        with_nms_path.append(item)
        labels.append(item.replace(root_path, ""))
    else:
        without_nms_path.append(item)


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
    dataset_type = 'CocoDataset'
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
    test = dict(
        type=dataset_type,
        classes=('person', 'car', 'truck', 'rider', 'bicycle', 'motorcycle',
                 'bus', 'train'),
        ann_file=path,
        img_prefix=val_split_path,
        pipeline=test_pipeline,
        test_mode=True)
    dataset = build_dataset(test)
    return dataset.coco.imgToAnns, dataset


def sup_norm(sup_list, state):
    if state == "sigmoid":
        return 1. / (1. + np.exp(-sup_list))
    elif state == "MaxMin":
        return 1. / sup_list


def match_gt_pd(best_pd, pd_woNMS, gt, COCOdataset):
    """match the ground truth to predict bounding box

    Args:
        best_pd (_type_): the predict bbox by nms
        pd_woNMS (_type_): all bboxes
        gt (_type_): ground truth

    Returns:
        _type_: _description_
    """
    # 0. prepare data
    best_pd_count = 0
    best_count = 0
    sup_count = 0
    metric = COCOdataset.evaluate(best_pd)
    results = []
    item = None

    for counts, (image_id, v) in enumerate(gt.items()):
        idkey = COCOdataset.img_ids.index(image_id)
        # image_id = imgInf.get("id")
        gt_list = gt[image_id]
        best_pd_list = best_pd[idkey]
        pd_woNMS_list = pd_woNMS[idkey]

        # for-bboxes (of a single image's gts), the output is gt_bboxes[cls_number][gt_number][bbox_xyxy]
        for j in range(len(gt_list)):
            gt_item = gt_list[j]

            gt_item['category_id'] = COCOdataset.cat2label.get(gt_item['category_id'])
            if gt_item['category_id'] == None:
                continue

            gt_bbox_mid = gt_item['bbox']
            gt_bbox_mid = np.array([gt_bbox_mid[0], gt_bbox_mid[1],
                                    gt_bbox_mid[0] + gt_bbox_mid[2], gt_bbox_mid[1] + gt_bbox_mid[3]])
            item_gt = np.array([gt_bbox_mid])
            # get the best_bboxes and the sup_bboxes according to class
            item_best_pd_list = np.array(best_pd_list[gt_item['category_id']][..., 0:4])
            item_best_pd_socre_list = np.array(best_pd_list[gt_item['category_id']][..., -1])
            item_sup_pd_list = np.array(pd_woNMS_list[gt_item['category_id']][..., 0:4])
            item_sup_pd_socre_list = np.array(pd_woNMS_list[gt_item['category_id']][..., -1])
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
                    item = {"image_id": image_id, "gt_bboxes": item_gt, "category_id": gt_item['category_id'],
                            "best_pd_bbox": np.array([]),
                            "best_pd_bbox_scores": np.array([0.0]),
                            "best_pd_bbox_IoU": np.array([0.0]),
                            "best_bbox": np.array([]),
                            "best_bbox_scores": np.array([0.0]),
                            "best_bbox_IoU": np.array([0.0]),
                            "sup_bboxes_06": np.array([]),
                            "sup_bboxes_06_scores": np.array([0.0]),
                            "sup_bboxes_06_IoU": np.array([0.0])}
                else:
                    item = {"image_id": image_id, "gt_bboxes": item_gt, "category_id": gt_item['category_id'],
                            "best_pd_bbox": np.array([]),
                            "best_pd_bbox_scores": np.array([0.0]),
                            "best_pd_bbox_IoU": np.array([0.0]),
                            "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                            "best_bbox_scores": np.array([item_sup_pd_socre_list[supbest_indices]]),
                            "best_bbox_IoU": np.array(supbest_value.numpy()),
                            "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                            "sup_bboxes_06_scores": np.array([item_sup_pd_socre_list[IoU_sup_torch_keep.numpy()]]),
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
                    item = {"image_id": image_id, "gt_bboxes": item_gt, "category_id": gt_item['category_id'],
                            "best_pd_bbox": np.array([]),
                            "best_pd_bbox_scores": np.array([0.0]),
                            "best_pd_bbox_IoU": np.array([0.0]),
                            "best_bbox": np.array([]),
                            "best_bbox_scores": np.array([0.0]),
                            "best_bbox_IoU": np.array([0.0]),
                            "sup_bboxes_06": np.array([]),
                            "sup_bboxes_06_scores": np.array([0.0]),
                            "sup_bboxes_06_IoU": np.array([0.0])}
                elif best_value < 0.5 and supbest_value >= 0.5:
                    item = {"image_id": image_id, "gt_bboxes": item_gt, "category_id": gt_item['category_id'],
                            "best_pd_bbox": np.array([]),
                            "best_pd_bbox_scores": np.array([0.0]),
                            "best_pd_bbox_IoU": np.array([0.0]),
                            "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                            "best_bbox_scores": np.array([item_sup_pd_socre_list[supbest_indices]]),
                            "best_bbox_IoU": np.array(supbest_value.numpy()),
                            "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                            "sup_bboxes_06_scores": np.array([item_sup_pd_socre_list[IoU_sup_torch_keep.numpy()]]),
                            "sup_bboxes_06_IoU": np.array(IoU_sup_torch[IoU_sup_torch_keep.numpy()].numpy())}
                    best_count += 1
                else:
                    # get coordinates
                    item = {"image_id": image_id, "gt_bboxes": item_gt, "category_id": gt_item['category_id'],
                            "best_pd_bbox": np.array([item_best_pd_list[best_indices]]),
                            "best_pd_bbox_scores": np.array([item_best_pd_socre_list[best_indices]]),
                            "best_pd_bbox_IoU": np.array(best_value.numpy()),
                            "best_bbox": np.array([item_sup_pd_list[supbest_indices]]),
                            "best_bbox_scores": np.array([item_sup_pd_socre_list[supbest_indices]]),
                            "best_bbox_IoU": np.array(supbest_value.numpy()),
                            "sup_bboxes_06": item_sup_pd_list[IoU_sup_torch_keep.numpy()],
                            "sup_bboxes_06_scores": np.array([item_sup_pd_socre_list[IoU_sup_torch_keep.numpy()]]),
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

    file = open('Cityscapesexperiments/test_Cityscapes.txt', 'w')
    # file.write('best_pd_mean', 'best_pd_std', 'best_mean', 'best_std', 'sup_mean', 'sup_std', 'label')
    file.write('best_pd_mean best_pd_std best_mean best_std sup_mean sup_std label\n')
    for i in range(len(with_nms_path)):
        gtAnns, COCOdataset = readgt(gt_json_path)
        best_pd = readpkl(with_nms_path[i])
        woNMS_pd = readpkl(without_nms_path[i])
        # label = os.path.basename(labels[i])
        labeli = os.path.basename(labels[i])
        label = labels[i]

        match_results, best_pd_count, best_count, sup_count = match_gt_pd(best_pd, woNMS_pd, gtAnns, COCOdataset)
        best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list = \
            data_trans(match_results, mode=["worst_edges", "IoU"], state="MaxMin")

        best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std = \
            fit_gaussian_2(best_pd_x_list, best_pd_y_list, best_x_list, best_y_list, sup_x_list, sup_y_list)
        print(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std, label, file=file)
        print((abs(best_pd_mean) + abs(best_pd_std)) / best_pd_count * 1000,
              (abs(best_mean) + abs(best_std)) / best_count * 1000,
              (abs(sup_mean) + abs(sup_std)) / sup_count * 1000000, file=file)
        print(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std, label)
        print(best_pd_count, best_count, sup_count)
        print((abs(best_pd_mean) + abs(best_pd_std))/best_pd_count*1000,
              (abs(best_mean) + abs(best_std))/best_count*1000,
              (abs(sup_mean) + abs(sup_std))/sup_count*1000000)
        # path = os.path.join('Cityscapesexperiments/plot', f'{labeli}.png')
        # draw_gaussian(best_pd_mean, best_pd_std, best_mean, best_std, sup_mean, sup_std, mode=["worst_edges", "IoU"],
        #               path=path)

    return


if __name__ == '__main__':
    main()
