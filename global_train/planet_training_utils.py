#!/usr/bin/env python

"""planet_training_utils.py

__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
import sys
import time

import cv2
import gdal
import h5py
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class PlanetScopeDataset_Prod(torch.utils.data.Dataset):
    def __init__(
        self, rootDir, status="train", patchSize=None, step=None, augmentation=True
    ):
        super(PlanetScopeDataset_Prod, self).__init__()

        self.status = status

        if self.status == "train":
            self.dataDir = os.path.join(rootDir, "train/img")
            self.labelDir = os.path.join(rootDir, "train/label_tsd")
            self.nameFiles = list_all_files(self.dataDir)
            self.numFiles = len(self.nameFiles)
        elif self.status == "val":
            self.dataDir = os.path.join(rootDir, "val/img")
            self.labelDir = os.path.join(rootDir, "val/label_tsd")
            self.nameFiles = list_all_files(self.dataDir)
            self.numFiles = len(self.nameFiles)
        else:
            self.dataName = rootDir
            self.img_shape, self.img = load_imageTest(self.dataName)
            self.patchSize = patchSize
            self.step = step

    def __len__(self):
        if self.status == "train":
            loop_len = self.numFiles
        elif self.status == "val":
            loop_len = self.numFiles
        else:
            loop_len = count_sliding_windows(self.img_shape, self.patchSize, self.step)

        return loop_len

    def __getitem__(self, i):
        if self.status == "train":
            data_p, label_p = load_image_tif_prod(
                self.nameFiles[i], self.dataDir, labelDir=self.labelDir
            )
        elif self.status == "val":
            data_p, label_p = load_image_tif_prod(
                self.nameFiles[i], self.dataDir, labelDir=self.labelDir
            )
        else:
            data_p, label_p = load_patch(
                self.img,
            )

        return (torch.from_numpy(data_p), torch.from_numpy(label_p))


def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)

    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)

    return _files


def load_image_tif(filename, dataDir, labelDir=None):
    fileName, fileExt = os.path.splitext(filename)
    tifName = os.path.join(dataDir, fileName) + ".tif"
    cityName = fileName.split("/")

    imgFID = gdal.Open(tifName)
    data_p = imgFID.ReadAsArray().astype(np.float32)

    if labelDir is not None:
        imgName = labelDir + "/" + cityName[-2] + "/" + cityName[-1] + ".png"
        label_p = cv2.imread(imgName, 0)

        return data_p, label_p


def load_image_tif_prod(filename, dataDir, labelDir=None):
    imgFID = gdal.Open(filename)
    data_p = imgFID.ReadAsArray().astype(np.float32)

    if labelDir is not None:
        x = filename.replace(".tif", ".png")
        imgName = x.replace("img", "label_tsd")
        label_p = cv2.imread(imgName, 0)

        # print(filename)
        # print(imgName)
        return data_p, label_p


def load_image_h5(filename, dataDir, labelDir=None):
    fileName, fileExt = os.path.splitext(filename)
    h5Name = os.path.join(dataDir, fileName) + ".h5"
    fid = h5py.File(h5Name, "r")
    data_p = fid["data"][:]
    data_p = np.float32(np.transpose(data_p, (2, 0, 1)))
    fid.close()

    # data_p = cv2.imread(os.path.join(dataDir, filename))

    if labelDir is not None:
        imgName = os.path.join(labelDir, fileName) + ".png"
        label_p = cv2.imread(imgName, 0)

    return data_p, label_p


def metrics(target, prediction, numClass):
    """
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    """

    cm = confusion_matrix(target, prediction, range(numClass))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print(f"{total} pixels processed")
    print(f"Total accuracy : {accuracy}%")

    print("---")

    # Compute F1 score
    F1Score = np.zeros(numClass)
    for i in range(numClass):
        try:
            F1Score[i] = 2.0 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print(f"{l_id}: {score}")

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))


def accuracy_binary(preds, labels, thres=5):
    preds_met = preds.clone().detach()
    labels_met = labels.clone().detach()
    preds_met[preds_met < thres] = 0
    preds_met[preds_met >= thres] = 1
    labels_met[labels_met < thres] = 0
    labels_met[labels_met >= thres] = 1

    acc = (preds_met == labels_met).sum()
    return acc


def iou_binary(preds, labels, thres=5):
    preds_met = preds.clone().detach()
    labels_met = labels.clone().detach()
    preds_met[preds_met < thres] = 0
    preds_met[preds_met >= thres] = 1
    labels_met[labels_met < thres] = 0
    labels_met[labels_met >= thres] = 1

    intersection = (preds_met * labels_met).sum()
    tmp = preds_met + labels_met
    tmp[tmp > 0] = 1
    union = tmp.sum()
    return intersection, union


def accuracy(preds, label):
    valid = label >= 0
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()
    imPred += 1
    imLab += 1
    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass)
    )
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f
