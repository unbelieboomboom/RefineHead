# code
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import math

from astropy.modeling import models, fitting


# gaussian fit
def astropy_gaussian(x, y, mean=0., std=10.0):
    g_init = models.Gaussian1D(amplitude=1., mean=mean, stddev=std)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y, maxiter=300)
    return g.mean.value, g.stddev.value

# global
Model_list = ["ATSS", "DyHead", "RefineHead", "YOLOv3",
              "YOLOX"]
Jtype = ["IoU05", "top10"]
Jimage = ["frankfurt_000001_039895_leftImg8bit"]
# path of the .json files
Save_dir = ""

def make_json_path(model_list, save_dir, jtype, jimage, name):
    json_path = []
    for model_name in model_list:
        for jtype_n in jtype:
            for jimage_n in jimage:
                for backbone_n in name:
                    json_path_item = os.path.join(save_dir, model_name)
                    json_path_item = os.path.join(json_path_item, jtype_n)
                    json_path_item = os.path.join(json_path_item, jimage_n)
                    json_path_item = os.path.join(json_path_item, backbone_n)
                    json_path_item = json_path_item + ".json"
                    json_path.append(json_path_item)
    return json_path

def make_json_path2(model_list, save_dir):
    json_path = []
    for model_name in model_list:
        json_path_item = os.path.join(save_dir, model_name)
        json_path_item = json_path_item + ".json"
        json_path.append(json_path_item)
    return json_path


def json_read(json_path):
    json_list = []
    for json_item in json_path:
        with open(json_item, 'r') as file:
            json_data = json.load(file)
            json_list.append(json_data)

    return json_list


def sup_norm(sup_list, state):
    if state == "sigmoid":
        return 1. / (1. + np.exp(-sup_list))
    elif state == "MaxMin":
        return 1. / sup_list


def delta_IoU(pred, target, eps=1e-7):
    n, _ = pred.shape
    target = target.repeat(n, axis=0)
    # overlap
    lt = np.maximum(pred[:, :2], target[:, :2])
    rb = np.minimum(pred[:, 2:], target[:, 2:])
    wh = rb - lt
    overlap = wh[:, 0] * wh[:, 1]
    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union
    return 1.0 - ious


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


def draw_gaussian(gaussian_result):
    color = ["blue", "orange", "gray", "red", "yellow", "purple"]
    # g_mean + 2* g_std
    min_values = []
    max_values = []
    for item in gaussian_result:
        min_values.append(item["g_mean"]-2*item["g_std"]-2)
        max_values.append(item["g_mean"]+2*item["g_std"]+2)
    min_value = min(min_values)
    max_value = max(max_values)
    x = np.linspace(min_value, max_value, 20*int(max_value-min_value))
    for i, item in enumerate(gaussian_result):
        model_name, g_mean, g_std = item["model"], item["g_mean"], item["g_std"]
        if model_name == 'RefineHead*':
            model_name = "RefineHead"

        y = normal_distribution(x, g_mean, g_std)
        plt.plot(x, y, color[i], label=model_name)
        plt.axvline(x=g_mean, color="black", linestyle='--')
        # plt.text(g_mean+0.5, 0.1, "({},{})".format(np.round(g_mean, 2), np.round(g_std, 2)), fontsize=12)
        # if i == 1:
        #     break
    plt.axvline(x=0, color="green", label="Ground-truth")
    # plt.annotate()
    plt.xlabel("Δx", fontsize=18)
    plt.ylabel("1/Δy", fontsize=18)
    plt.legend(fontsize=14)
    plt.show()


def draw_gaussian_y(gaussian_result):
    color = ["blue", "orange", "gray", "red", "yellow", "purple"]
    x = np.linspace(-20, 50, 700)
    for i, item in enumerate(gaussian_result):
        model_name, g_mean, g_std = item["model"], item["g_mean"], item["g_std"]

        y = normal_distribution(x, g_mean, g_std)
        plt.plot(x, y, color[i], label=model_name)
        plt.axvline(x=g_mean, color="black", linestyle='--')

        if i == 1:
            plt.text(g_mean + 2, 0.21, "({},{})".format(int(np.round(g_mean)), int(np.round(g_std))), fontsize=12)
            break
        else:
            plt.text(g_mean - 9, 0.17, "({},{})".format(int(np.round(g_mean)), int(np.round(g_std))), fontsize=12)
    plt.axvline(x=0, color="green", label="Ground-truth")
    # plt.annotate()
    plt.xlabel("Δy_rb", fontsize=18)
    plt.ylabel("1/Δx_rb", fontsize=18)
    plt.legend(fontsize=14)
    plt.show()


def draw_gaussian2(gaussian_result):
    color = ["blue", "orange", "gray", "red", "yellow", "purple"]
    # g_mean + 2* g_std
    min_values = []
    max_values = []
    for item in gaussian_result:
        min_values.append(item["g_mean"] - 2 * item["g_std"] - 10)
        max_values.append(item["g_mean"] + 2 * item["g_std"] + 10)
    min_value = min(min_values)
    max_value = max(max_values)
    x = np.linspace(min_value, max_value, 200)
    for i, item in enumerate(gaussian_result):
        model_name, g_mean, g_std = item["model"], item["g_mean"], item["g_std"]
        if model_name == 'RefineHead*':
            model_name = "RefineHead"

        y = normal_distribution(x, g_mean, g_std)
        plt.plot(x, y, color[i], label=model_name)
        plt.axvline(x=g_mean, color="black", linestyle='--')

    plt.axvline(x=0, color="green", label="Ground-truth")
    # plt.annotate()
    plt.xlabel("Δx", fontsize=18)
    plt.ylabel("1/ΔIoU", fontsize=18)
    plt.legend(fontsize=14)
    plt.show()
    plt.close()

# only the rb corner point is considered here
def fit_gaussian(json_list):
    gaussian_results = []
    for item in json_list:
        item["sup_bboxes"] = item["sup_bboxes"][:3]
        # gt process

        x_gt = np.array(item["gt_bboxes"][0][2])
        y_gt = np.array(item["gt_bboxes"][0][3])
        # y_gt_reverse = 1000. / y_gt

        # bboxes process
        x_pred = np.array(item["bboxes"][0][2])
        y_pred = np.array(item["bboxes"][0][3])
        x_pred_reverse = x_gt / x_pred
        y_pred_reverse = y_gt / y_pred

        # sup_bboxes process
        x_sup = np.array(item["sup_bboxes"])[:, 2]
        y_sup = np.array(item["sup_bboxes"])[:, 3]
        delta_x_sup = x_sup - x_gt
        delta_y_sup = y_sup - y_gt
        y_sup_reverse = sup_norm(delta_y_sup, state="MaxMin")
        # fit gaussian_curves
        g_mean, g_std = astropy_gaussian(delta_x_sup, y_sup_reverse, mean=0.0, std=10.0)
        gaussian_result = {
            "model": item["model"],
            "g_mean": g_mean,
            "g_std": g_std
        }
        gaussian_results.append(gaussian_result)
    return gaussian_results


# only the rb corner point is considered here
def fit_gaussian_y(json_list):
    gaussian_results = []
    for item in json_list:
        # gt process

        x_gt = np.array(item["gt_bboxes"][0][2])
        y_gt = np.array(item["gt_bboxes"][0][3])
        # y_gt_reverse = 1000. / y_gt

        # bboxes process
        x_pred = np.array(item["bboxes"][0][2])
        y_pred = np.array(item["bboxes"][0][3])
        x_pred_reverse = x_gt / x_pred
        y_pred_reverse = y_gt / y_pred

        # sup_bboxes process
        x_sup = np.array(item["sup_bboxes"])[:, 2]
        y_sup = np.array(item["sup_bboxes"])[:, 3]
        delta_x_sup = x_sup - x_gt
        delta_y_sup = y_sup - y_gt
        x_sup_reverse = sup_norm(delta_x_sup, state="MaxMin")
        # fit gaussian_curves
        g_mean, g_std = astropy_gaussian(delta_y_sup, x_sup_reverse, mean=0.0, std=10.0)
        gaussian_result = {
            "model": item["model"],
            "g_mean": g_mean,
            "g_std": g_std
        }
        gaussian_results.append(gaussian_result)
    return gaussian_results


# x is center point, y is Csup/gt
def fit_gaussian_2(json_list, x_mode=True):
    gaussian_results = []
    for item in json_list:
        # gt process
        item["sup_bboxes"] = item["sup_bboxes"][:3]

        x_gt = np.array(item["gt_bboxes"])[0][0::2]
        y_gt = np.array(item["gt_bboxes"])[0][1::2]
        center_x_gt = np.mean(x_gt)
        center_y_gt = np.mean(y_gt)
        # y_gt_reverse = 1000. / y_gt

        # bboxes process
        x_pred = np.array(item["bboxes"])[0][0:4][0::2]
        y_pred = np.array(item["bboxes"])[0][1::2]
        center_x_pred = np.mean(x_pred)
        center_y_pred = np.mean(y_pred)

        # sup_bboxes process
        x_sup = np.array(item["sup_bboxes"])[:, 0:4][:, 0::2]
        y_sup = np.array(item["sup_bboxes"])[:, 1::2]
        center_x_sup = np.mean(x_sup, axis=-1)
        center_y_sup = np.mean(y_sup, axis=-1)

        delta_x_sup = center_x_sup - center_x_gt
        delta_y_sup = center_y_sup - center_y_gt
        delta_IoU_sup = delta_IoU(np.array(item["sup_bboxes"])[:, 0:4], np.array(item["gt_bboxes"]))
        IoU_sup_reverse = sup_norm(delta_IoU_sup, state="MaxMin")
        # fit gaussian_curves
        if x_mode is True:
            g_mean, g_std = astropy_gaussian(delta_x_sup, IoU_sup_reverse, mean=0.0, std=10.0)
        else:
            x_gt = np.array(item["gt_bboxes"][0][2])
            y_gt = np.array(item["gt_bboxes"][0][3])
            x_sup = np.array(item["sup_bboxes"])[:, 2]
            y_sup = np.array(item["sup_bboxes"])[:, 3]
            delta_x_sup = x_sup - x_gt
            delta_y_sup = y_sup - y_gt
            g_mean, g_std = astropy_gaussian(delta_x_sup, IoU_sup_reverse, mean=0.0, std=10.0)

        gaussian_result = {
            "model": item["model"],
            "g_mean": g_mean,
            "g_std": g_std
        }
        gaussian_results.append(gaussian_result)
    return gaussian_results


# only the rb corner point is considered here
def draw_scatter(json_list):
    color_list = []
    for i, item in enumerate(json_list):
        # gt process

        x_gt = np.array(item["gt_bboxes"][0][2])
        y_gt = np.array(item["gt_bboxes"][0][3])
        # y_gt_reverse = 1000. / y_gt

        # bboxes process
        x_pred = np.array(item["bboxes"][0][2])
        y_pred = np.array(item["bboxes"][0][3])
        # x_pred_reverse = x_gt / x_pred
        # y_pred_reverse = y_gt / y_pred

        # sup_bboxes process
        x_sup = np.array(item["sup_bboxes"])[:, 2]
        y_sup = np.array(item["sup_bboxes"])[:, 3]
        # remove x_pred from x_sup
        remove_index_x = x_sup == x_pred
        x_sup = x_sup[~remove_index_x]
        y_sup = y_sup[~remove_index_x]
        delta_x_sup = x_sup
        delta_y_sup = y_sup - y_gt

        plt.scatter(x_sup, y_sup, s=60, marker='o', c="blue", label="Top 10 predictions")
        plt.scatter(x_gt, y_gt, s=100, marker='*', c="green", label="Ground-truth")
        plt.scatter(x_pred, y_pred, s=100, marker='P', c="red", label="The Best bbox")

        plt.xlim(1182, 1212)
        plt.ylim(487, 503)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(plt.MultipleLocator(3))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(3))
        plt.legend(fontsize=17)
        plt.annotate(text="({}, {})".format(int(x_gt), int(y_gt)), xy=(x_gt-3.3, y_gt+0.4), fontsize=15)

        plt.show()
    plt.close()


def main(pltguassian=True, pltscatter=False):
    # json_path = make_json_path(model_list=Model_list, save_dir=Save_dir, jtype=["IoU05"], jimage=Jimage, name=["R-50", "R-101"])
    json_path = make_json_path2(model_list=["ATSS", "DyHead",
                                            "YOLOX", "RefineHead+"], save_dir=Save_dir2)
    # json_path = make_json_path2(model_list=["ATSS"], save_dir=Save_dir2)
    json_list = json_read(json_path)
    if pltguassian is True:
        gaussian_results = fit_gaussian(json_list)
        draw_gaussian(gaussian_results)
        gaussian_results = fit_gaussian_2(json_list, x_mode=False)
        draw_gaussian2(gaussian_results)
    if pltscatter is True:
        draw_scatter(json_list[0:2])

    return


if __name__ == '__main__':
    main(pltguassian=True, pltscatter=False)



