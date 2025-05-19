#!/usr/bin/env python

"""planet_training_demo.py

__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
import timeit

import torch
from models.fc_densenet.tiramisu_de import FCDenseNet67
from planet_training_utils import (
    PlanetScopeDataset_Prod,
    accuracy_binary,
    iou_binary,
    progress_bar,
)
from torch.utils.data import DataLoader

# from models.efficientunet.efficientunet import *
# from models.efficientunet_ggcn.efficientunet_ggcn import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")

# Parameters
bS = 16
pS = 256
numP = 256 * 256
numChannels = 4
numClass = 11
numW = 2
valComp = -10e10

# Dataset
dataDir = "your_data_directory"
trainDataset = PlanetScopeDataset_Prod(dataDir, status="train")
trainLoader = DataLoader(
    dataset=trainDataset, num_workers=numW, pin_memory=True, batch_size=bS, shuffle=True
)
valDataset = PlanetScopeDataset_Prod(dataDir, status="val")
valLoader = DataLoader(
    dataset=valDataset, num_workers=numW, pin_memory=True, batch_size=bS, shuffle=False
)
batch_total = len(trainLoader)
idx_train_total = len(trainDataset)
idx_val_total = len(valDataset)

# Set loss and model
pretrained = True
model_checkpoint_load = "./your_check_point_for_load"
model_checkpoint = "./your_check_point"
model = FCDenseNet67(in_channels=numChannels, n_classes=numClass)
# model = get_efficientunet_ggcn_b0(out_channels=numClass, concat_input=True, pretrained=False, in_channels=numChannels)
# model = get_efficientunet_b0(out_channels=numClass, concat_input=True, pretrained=False, in_channels=numChannels)
model = model.to(device)

logger_checkpoint = "./your_logs"
loggerFID = open(logger_checkpoint, "w")

if pretrained:
    model_pretrained_checkpoint = model_checkpoint_load
    checkpoint = torch.load(model_pretrained_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochS = checkpoint["epoch"]
    trainLoss = checkpoint["loss"]
else:
    epochS = 0
    trainLoss = 0.0

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

if pretrained:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

loss_func = torch.nn.NLLLoss()
loss_func = loss_func.to(device)

tsdThres = 5

# Train
for epoch in range(epochS, 1000):
    print(f"epoch {epoch + 1}")

    model.train()
    trainLoss = 0.0
    trainAcc2 = 0.0
    trainAcc11 = 0.0
    train_startTime = timeit.default_timer()

    for batch_idx, (dataB, labelB) in enumerate(trainLoader, 1):
        dataB, labelB = dataB.to(device), labelB.to(device)
        optimizer.zero_grad()
        outB = model(dataB)
        predB = torch.max(outB, 1)[1]
        trainAcc11B = (predB == labelB).sum()
        trainAcc2B = accuracy_binary(predB, labelB, thres=tsdThres)
        trainAcc11 += trainAcc11B.item()
        trainAcc2 += trainAcc2B.item()
        loss = loss_func(outB, labelB.long())
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar(
            batch_idx,
            batch_total,
            "Loss: %.3f | Acc: %.3f | Acc2: %.3f%% (%d/%d)"
            % (
                trainLoss / batch_idx,
                100.0 * trainAcc11 / idx_train_total / numP,
                100.0 * trainAcc2 / idx_train_total / numP,
                batch_idx * bS,
                idx_train_total,
            ),
        )

    model.eval()
    valLoss = 0.0
    valAcc2 = 0.0
    valAcc11 = 0.0
    valInter = 0.0
    valUnion = 0.0
    val_startTime = timeit.default_timer()

    with torch.no_grad():
        for batch_idx, (dataB, labelB) in enumerate(valLoader):
            dataB, labelB = dataB.to(device), labelB.to(device)
            outB = model(dataB)
            predB = torch.max(outB, 1)[1]
            valAcc11B = (predB == labelB).sum()
            valAcc2B = accuracy_binary(predB, labelB, thres=tsdThres)
            valAcc11 += valAcc11B.item()
            valAcc2 += valAcc2B.item()
            valInterB, valUnionB = iou_binary(predB, labelB, thres=tsdThres)
            valInter += valInterB.item()
            valUnion += valUnionB.item()
            loss = loss_func(outB, labelB.long())
            valLoss += loss.item()

        if (valInter / valUnion) > valComp:
            valComp = valInter / valUnion
            bestAcc2 = valAcc2 / idx_val_total / numP
            bestAcc11 = valAcc11 / idx_val_total / numP
            bestIoU = valInter / valUnion
            bestEpoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": trainLoss,
                },
                model_checkpoint,
            )

    val_endTime = timeit.default_timer()
    print(
        f"Test Loss: {valLoss / idx_val_total:.4f}, Acc: {valAcc11 / idx_val_total / numP:.4f}, Acc2: {valAcc2 / idx_val_total / numP:.4f}, IoU: {valInter / valUnion:.4f}, Time: {val_endTime - val_startTime:.3f} | BestIoU: {bestIoU:.4f}, Acc: {bestAcc11:.4f}, Acc2: {bestAcc2:.4f}, IoU: {bestIoU:.4f}, Epoch: {bestEpoch:d}"
    )

    loggerFID.write(
        f"{epoch + 1} {trainLoss / idx_train_total:.6f} {trainAcc11 / idx_train_total / numP:.6f} {trainAcc2 / idx_train_total / numP:.6f} {valLoss / idx_val_total:.6f} {valAcc11 / idx_val_total / numP:.6f} {valAcc2 / idx_val_total / numP:.6f} {valInter / valUnion:.6f}\n"
    )
    loggerFID.flush()

loggerFID.close()
