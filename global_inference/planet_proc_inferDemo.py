#!/usr/bin/env python

"""planet_inferencing_inferDemo.py: demo processing with Planet data.

__copyright__   :  "Copyright 2020, The So2Sat Project"
__version__     :  "1.0.0"
__status__      :  "Production"
__last_update__ :  "02.08.2020"

"""

import os
import numpy as np
import pandas as pd

from planet_inferencing_sr_gpu import *

# setting parameters
#continents = ['africa', 'southamerica', 'oceania', 'asiaeast', 'asiawest', 'nordamerica', 'europe']

prodType = 'SR'
modelType = 'efficientunet'
normType = 1

if (modelType == 'fcdense'):
    if ((prodType == 'SR') and (normType == 1)):
        modelFilename = '../models/fcdensenet67_sr.pth'
    if ((prodType == 'BM') and (normType == 1)):
        modelFilename = '../models/fcdensenet67_bm.pth'
    if ((prodType == 'SR') and (normType == 2)):
        modelFilename = '../models/fcdensenet67_sr_patch.pth'
    if ((prodType == 'BM') and (normType == 2)):
        modelFilename = '../models/fcdensenet67_bm_patch.pth'
        
if (modelType == 'efficientunet'):
    modelSuffix = '_EB0'
    if (prodType == 'SR'):
        modelFilename = '/data/shi/planet/global/models/efficientunetb0_sr.pth'
    else:
        modelFilename = '/data/shi/planet/global/models/efficientunetb0_bm.pth'

if (modelType == 'ggcn'):
    modelSuffix = '_GGCN'
    if (prodType == 'SR'):
        modelFilename = '/data/shi/planet/global/models/ggcn_fcdensenet67_sr.pth'
    else:
        modelFilename = '/data/shi/planet/global/models/ggcn_fcdensenet67_bm.pth'

gpuRank = 1
satImg = 'your_satellite_image.tif'
predDir = 'your_prediction_directory'
predFile = 'your_prediction_filename'
planet_infer_sliding_window(gpuRank, modelFilename, satImg, predDir, predFile, normType)
