#!/usr/bin/env python

"""planet_inferencing_utils.py: utilities for inferencing with Planet Data.

__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import itertools
import os
import sys
import time

import gdal
import numpy as np


def planet_infer_getDirs(rootDir):
    DirNames = [
        d for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, d))
    ]
    return sorted(DirNames)


def planet_infer_listAllFiles(rootdir):
    _files = []
    dirList = os.listdir(rootdir)
    for i in range(0, len(dirList)):
        pathList = os.path.join(rootdir, dirList[i])
        if os.path.isdir(pathList):
            _files.extend(planet_infer_listAllFiles(pathList))
        if os.path.isfile(pathList):
            _files.append(pathList)
    return _files


def planet_infer_normalize_ahcUpSolo(filename, iqrC):
    img = planet_infer_readTiff(filename)

    imgStat = 1

    if filename.find("_bm_mosaic_tile.tif") != -1:
        img = np.float32(img[:3, :, :])

    imgC, imgH, imgW = img.shape
    # print(img.shape)
    imgOut = np.zeros((imgC, imgH, imgW), dtype=np.float32)

    for nC in range(imgC):
        data = img[nC, :, :]
        data_new = data
        countzero = np.count_nonzero(data)
        if countzero == 0:
            imgStat = 0
        else:
            q1 = np.percentile(data[data > 0], 25)
            q3 = np.percentile(data[data > 0], 75)
            iqr = q3 - q1
            clipMax = q3 + iqr * iqrC
            data_new[data_new > clipMax] = clipMax
            data_new = data_new / clipMax
            imgOut[nC, :, :] = data_new

    return imgOut, imgStat


def planet_infer_normalize_ahcUpSolo_patch(filename, iqrC, pS=512):
    img = planet_infer_readTiff(filename)

    if filename.find("_bm_mosaic_tile.tif") != -1:
        img = np.float32(img[:3, :, :])

    imgStat = 1

    imgC, imgH, imgW = img.shape
    imgOut = np.zeros((imgC, imgH, imgW), dtype=np.float32)

    for idP, coords in enumerate(
        planet_infer_grouper(
            1, planet_infer_swCoords(img, step=pS, window_size=(pS, pS))
        )
    ):
        x, y, w, h = coords[0]
        imgP = img[:, x : x + w, y : y + h]
        for nC in range(imgC):
            data = imgP[nC, :, :]
            data_new = data
            countzero = np.count_nonzero(data)
            if countzero == 0:
                imgStat = 0
            else:
                q1 = np.percentile(data[data > 0], 25)
                q3 = np.percentile(data[data > 0], 75)
                iqr = q3 - q1
                clipMax = q3 + iqr * iqrC
                data_new[data_new > clipMax] = clipMax
                data_new = data_new / clipMax
                imgOut[nC, x : x + w, y : y + h] = data_new

    return imgOut, imgStat


def planet_infer_readTiff(tiffFilename):
    imgFID = gdal.Open(tiffFilename)
    img = imgFID.ReadAsArray()

    return img


def planet_infer_writeTiff(inTiff, outTiff, outData, dataType):
    inDs = gdal.Open(inTiff)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize

    if dataType == "uint32":
        outDs = gdal.GetDriverByName("GTiff").Create(
            outTiff, cols, rows, 1, gdal.GDT_UInt32
        )
    else:
        outDs = gdal.GetDriverByName("GTiff").Create(
            outTiff, cols, rows, 1, gdal.GDT_Byte
        )

    outband = outDs.GetRasterBand(1)
    outband.WriteArray(outData, 0, 0)
    outband.SetNoDataValue(0)

    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    outband.FlushCache()


def planet_infer_swCoords(img, step=128, window_size=(256, 256)):
    H = img.shape[1]
    W = img.shape[2]

    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, H, step):
        if x + window_size[0] > H:
            x = H - window_size[0]
        for y in range(0, W, step):
            if y + window_size[1] > W:
                y = W - window_size[1]
            yield x, y, window_size[0], window_size[1]


def planet_infer_swCount(img, step=128, window_size=(256, 256)):
    H = img.shape[1]
    W = img.shape[2]

    """ Count the number of windows in an image """
    nSW = 0
    for x in range(0, H, step):
        if x + window_size[0] > H:
            x = H - window_size[0]
        for y in range(0, W, step):
            if y + window_size[1] > W:
                y = W - window_size[1]
            nSW += 1

    return nSW


def planet_infer_grouper(n, iterable):
    """Browse an iterator by chunk of n elements"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(cm):
    UAur = float(cm[1][1]) / float(cm[1][0] + cm[1][1])
    # UAnonur = float(cm[0][0]) / float(cm[0][0] + cm[0][1])
    PAur = float(cm[1][1]) / float(cm[0][1] + cm[1][1])
    # PAnonur = float(cm[0][0]) / float(cm[1][0] + cm[0][0])
    OA = float(cm[1][1] + cm[0][0]) / float(cm[1][0] + cm[1][1] + cm[0][0] + cm[1][0])
    F1 = 2 * UAur * PAur / (UAur + PAur)
    IoU = float(cm[1][1]) / float(cm[1][0] + cm[1][1] + cm[0][1])

    return OA, F1, IoU


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
