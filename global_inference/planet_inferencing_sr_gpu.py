#!/usr/bin/env python

"""planet_inferencing_sr_gpu.py: inferencing on GPU with Planet data.

__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
import timeit

import cv2
import numpy as np
import torch
from networks.efficientunet.efficientunet import get_efficientunet_b0
from networks.fc_densenet.tiramisu_de import FCDenseNet67
from networks.fc_densenet_ggcn.tiramisu_ggcn import FCDenseNet67_ggcn
from planet_inferencing_utils import (
    planet_infer_grouper,
    planet_infer_normalize_ahcUpSolo,
    planet_infer_normalize_ahcUpSolo_patch,
    planet_infer_swCoords,
    planet_infer_swCount,
    planet_infer_writeTiff,
    progress_bar,
)


def planet_infer_sliding_window(
    gpuRank, modelFilename, satImg, predDir, predFile, normType=1
):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuRank)

    bS = 32
    tsdThres = 5
    numClass = 11
    numChannels = 4

    if not os.path.exists(predDir):
        os.makedirs(predDir)

    if modelFilename.find("efficientunet") != -1:
        bS = 128
    elif modelFilename.find("ggcn") != -1:
        bS = 16
    else:
        bS = 32

    bfsPNG = predDir + predFile + "_sr_ss.png"
    bfsTIF = predDir + predFile + "_sr_ss.tif"

    if modelFilename.find("_sr_tsd_") != -1:
        numChannels = 4
    else:
        numChannels = 3
        bfsPNG = predDir + predFile + "_bm_ss.png"
        bfsTIF = predDir + predFile + "_bm_ss.tif"

    if normType == 1:
        img, imgStat = planet_infer_normalize_ahcUpSolo(satImg, 1.5)
    else:
        img, imgStat = planet_infer_normalize_ahcUpSolo_patch(satImg, 1.5, pS=256)

    if imgStat:
        if modelFilename.find("efficientunet") != -1:
            model = get_efficientunet_b0(
                out_channels=numClass,
                concat_input=True,
                pretrained=False,
                in_channels=numChannels,
            )
        elif modelFilename.find("ggcn") != -1:
            model = FCDenseNet67_ggcn(in_channels=numChannels, n_classes=numClass)
        else:
            model = FCDenseNet67(in_channels=numChannels, n_classes=numClass)

        checkpoint = torch.load(modelFilename)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.cuda()
        model.eval()

        inference_startTime = timeit.default_timer()

        pred = np.zeros(img.shape[1:] + (numClass,), dtype=np.float32)
        numP = planet_infer_swCount(img)

        with torch.no_grad():
            for idB, coords in enumerate(
                planet_infer_grouper(
                    bS, planet_infer_swCoords(img, step=128, window_size=(256, 256))
                )
            ):
                imgP = [np.copy(img[:, x : x + w, y : y + h]) for x, y, w, h in coords]
                imgP = np.asarray(imgP)
                imgP = torch.from_numpy(imgP).float().cuda()
                outs = model(imgP)
                outs_np = outs.detach().cpu().numpy()

                for out, (x, y, w, h) in zip(outs_np, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x : x + w, y : y + h] += out

                progress_bar(idB, numP, "numBatch: (%d/%d)" % ((idB + 1), numP))

        pred = np.argmax(pred, axis=-1)
        pred = np.where(pred > tsdThres, 255, 0)
        cv2.imwrite(bfsPNG, pred.astype(np.uint8))
        planet_infer_writeTiff(satImg, bfsTIF, pred, "uint8")

        inference_endTime = timeit.default_timer()
        print("-- inferencing : " + str(inference_endTime - inference_startTime) + " s")
