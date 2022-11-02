import time
import torch
import os
import json
import numpy as np


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s))


def dfs_search(pre_path, now_path):
    path = os.path.join(pre_path, now_path)
    file_list = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_list = file_list + dfs_search(pre_path, os.path.join(now_path, filename))
    else:
        file_list = [path]

    return file_list


def get_file_list(file_path, file_name):
    real_file_list = []
    file_list = file_name.replace(" ", "").split(",")
    for a in range(0, len(file_list)):
        real_file_list = real_file_list + dfs_search(file_path, file_list[a])
    return real_file_list


def check_multi(config):
    multi_label = ["multi_label_cross_entropy_loss"]
    if config.get("train", "type_of_loss") in multi_label:
        return True
    else:
        return False


def calc_accuracy(outputs, label, config, result=None):
    from utils.accuracy import top1, top2
    if config.get("output", "accuracy_method") == "top1":
        return top1(outputs, label, config, result)
    elif config.get("output", "accuracy_method") == "top2":
        return top2(outputs, label, config, result)
    else:
        raise NotImplementedError


def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_result(res, print=False):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    if print:
        print_info("Micro precision\t%.3f" % micro_precision)
        print_info("Macro precision\t%.3f" % macro_precision)
        print_info("Micro recall\t%.3f" % micro_recall)
        print_info("Macro recall\t%.3f" % macro_recall)
        print_info("Micro f1\t%.3f" % micro_f1)
        print_info("Macro f1\t%.3f" % macro_f1)



def get_IOU(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2
    FN = ((SR == 0) + (GT == 1)) == 2

    IOU = float(torch.sum(TP))/(float(torch.sum(TP+FP+FN)) + 1e-6)

    return IOU

# PyTroch version:https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()
