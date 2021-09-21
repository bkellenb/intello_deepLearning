'''
    Common utility functions.

    2021 Benjamin Kellenberger
'''

import os
import glob
import math
import copy
import numpy as np

import rasterio
import rasterio.features
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers import FrozenBatchNorm2d, NaiveSyncBatchNorm
from detectron2.structures.masks import polygons_to_bitmask
from detectron2.data.catalog import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


IMAGE_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.tif',
    '.tiff',
    '.png'
)


def listImages(baseDir, recursive=True):
    imgs = set(glob.glob(os.path.join(baseDir, '**/*'+IMAGE_EXTENSIONS[0]), recursive=recursive))
    for ie in IMAGE_EXTENSIONS[1:]:
        imgs = imgs.union(set(glob.glob(os.path.join(baseDir, '**/*'+ie), recursive=recursive)))
    return imgs



def loadImage(filePath, normalisation_value=1, makeUint8=False):
    with rasterio.open(filePath) as f:
        image = f.read().astype(np.float32)
        coords = next(rasterio.features.shapes(f.dataset_mask(), transform=f.transform))[0]['coordinates'][0]
    image = image / normalisation_value
    if makeUint8:
        image = (image * 255).astype(np.uint8)
    return image, coords, f.transform



def load_image_folder(directory):
    images = set()
    for ie in IMAGE_EXTENSIONS:
        images = images.union(set(glob.glob(os.path.join(directory, '**/*'+ie), recursive=True)))
        images = images.union(set(glob.glob(os.path.join(directory, '**/*'+ie.upper()))))
    
    # bring into minimal Detectron2-compliant form
    images = [{'image_id': idx, 'file_name': i, 'annotations': []} for idx, i in enumerate(images)]
    return images



def load_config_dataset(cfg, split='train'):
    '''
        Currently only supports COCO-formatted datasets.
    '''
    dsName = cfg.DATASETS.NAME + '_' + split
    register_coco_instances(dsName, {},
        os.path.join(cfg.DATASETS.DATA_ROOT, split+'.json'),
        cfg.DATASETS.DATA_ROOT)
    setattr(cfg.DATASETS, split.upper(), dsName)
    dataset = DatasetCatalog.get(dsName)

    # also register "stuff classes" for semantic segmentation models and mask background class
    metadata = MetadataCatalog.get(dsName)
    if hasattr(metadata, 'thing_classes') and not hasattr(metadata, 'stuff_classes'):
        stuff_classes = copy.deepcopy(metadata.thing_classes)
        stuff_classes.insert(0, 'background')
        MetadataCatalog.get(dsName).stuff_classes = stuff_classes

    return dataset, dsName



def saveImage(image, filePath, out_meta={}):
    if 'width' not in out_meta:
        out_meta['width'] = image.shape[2]
    if 'height' not in out_meta:
        out_meta['height'] = image.shape[1]
    if 'count' not in out_meta:
        out_meta['count'] = image.shape[0]
    if 'dtype' not in out_meta:
        out_meta['dtype'] = str(image.dtype)
    with rasterio.open(filePath, 'w', **out_meta) as dest_img:
        dest_img.write(image)



def _replaceBatchNorm(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        replClass = None
        if type(target_attr) == nn.BatchNorm1d:
            replClass = nn.InstanceNorm1d
        elif type(target_attr) == nn.BatchNorm2d or \
            type(target_attr) == FrozenBatchNorm2d or \
                type(target_attr) == NaiveSyncBatchNorm:
            replClass = nn.InstanceNorm2d
        elif type(target_attr) == nn.BatchNorm3d:
            replClass = nn.InstanceNorm3d
        
        if replClass is not None:
            replacement = replClass(target_attr.num_features, target_attr.eps,
                                    affine=False, track_running_stats=False)
            setattr(module, attr_str, replacement)

    # apply to children as well
    for _, immediate_child_module in module.named_children():
        _replaceBatchNorm(immediate_child_module)




def loadModel(cfg, resume=True, startIter=None):
    '''
        Performs the following steps:
        1. Build a base Detectron2 model, load pre-trained weights
        2. Modify number of input channels if needed to accommodate e.g. RGBNIR data
        3. Replace batch norm with instance norm layers if specified
    '''
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)

    # load pre-trained weights
    checkpointer = DetectionCheckpointer(model)
    try:
        checkpointer.load(cfg.MODEL.WEIGHTS)
    except Exception as e:
        print(f'Could not load pre-trained model weights ("{e}").')

    # modify model to accommodate a different number of input bands (if needed)         TODO: only works for Faster R-CNN and Mask R-CNN right now
    if cfg.INPUT.NUM_INPUT_CHANNELS != 3 and cfg.MODEL.META_ARCHITECTURE != 'UNet':
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

    # replace batch norm layers if needed
    try:
        replaceBatchNorm = cfg.MODEL.REPLACE_BATCH_NORM
    except:
        replaceBatchNorm = False
    if replaceBatchNorm:
        _replaceBatchNorm(model)


    # load existing model weights
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    if resume and checkpointer.has_checkpoint():
        states = checkpointer.get_all_checkpoint_files()
        if startIter is not None and startIter >= 0 and len(states):
            # find closest starting point
            parent, _ = os.path.split(states[0])
            baseName = os.path.join(parent, 'model_')
            closestState = None
            diff = 1e9
            for s in states:
                try:
                    epoch = int(s.replace(baseName, '').replace('.pth', ''))
                    newDiff = abs(epoch - startIter)
                    if newDiff < diff:
                        diff = newDiff
                        closestState = s
                except:
                    pass
            startIter = checkpointer.load(closestState).get('iteration', -1) + 1
            if diff > 1:
                print(f'Closest checkpoint to requested model iteration {startIter}: "{closestState}"')
        else:
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


def instances_to_segmask(instances, size, class_offset=0):
    '''
        Receives Detectron2-formatted instances and creates a semantic
        segmentation mask according to the specified size (tuple of W, H). Adds
        the indicated offset to the class ordinals. Note that any pixels where
        instances overlap will be assigned the last instance in order.
    '''
    segmask = torch.zeros(size, dtype=torch.long)
    for i in range(len(instances)):
        inst = instances[i]
        if hasattr(inst, 'gt_masks'):
            for p, poly in enumerate(inst.gt_masks.polygons):
                mask = polygons_to_bitmask(poly, height=size[0], width=size[1])
                segmask[mask] = inst.gt_classes[p] + class_offset
    return segmask
