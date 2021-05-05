'''
    Common utility functions.

    2021 Benjamin Kellenberger
'''

import os
import math
import numpy as np

import rasterio
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer



def loadModel(cfg):
    '''
        Performs the following steps:
        1. Build a base Detectron2 model, load pre-trained weights
        2. Modify number of input channels if needed to accommodate e.g. RGBNIR data
    '''
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)

    # load pre-trained weights
    checkpointer = DetectionCheckpointer(model)
    try:
        checkpointer.load(cfg.MODEL.WEIGHTS)
    except Exception as e:
        print(f'Could not load pre-trained model weights ("{e}").')

    # modify model to accommodate a different number of input bands (if needed)         TODO: only works for Mask R-CNN right now
    if cfg.INPUT.NUM_INPUT_CHANNELS != 3:
        # replicate weights and pixel mean and std
        numRep = math.ceil(cfg.INPUT.NUM_INPUT_CHANNELS / 3)
        weight = model.backbone.stem.conv1.weight
        if numRep > 1:
            weight = weight.repeat(1, numRep, 1, 1)
            model.pixel_mean = model.pixel_mean.repeat(numRep,1,1)
            model.pixel_std = model.pixel_std.repeat(numRep,1,1)
        weight = weight[:,:cfg.INPUT.NUM_INPUT_CHANNELS,:,:]
        model.backbone.stem.conv1.weight = torch.nn.Parameter(weight)
        model.backbone.stem.conv1.in_channels = cfg.INPUT.NUM_INPUT_CHANNELS
        model.pixel_mean = model.pixel_mean[:cfg.INPUT.NUM_INPUT_CHANNELS,...]
        model.pixel_std = model.pixel_std[:cfg.INPUT.NUM_INPUT_CHANNELS,...]

    # load existing model weights
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    if checkpointer.has_checkpoint():
        startIter = checkpointer.resume_or_load(checkpointer.get_checkpoint_file()).get('iteration', -1) + 1
    else:
        startIter = 0

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    # model.load_state_dict(torch.load("./model_final.pth", map_location='cpu'))
    return model, checkpointer, startIter



def loadImage(filePath):
    with rasterio.open(filePath) as f:
        image = f.read().astype(np.float32)
    image = image / 65535 * 255       #TODO
    return image