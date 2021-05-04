'''
    General-purpose routine for training Detectron2 models.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import math
import logging

from tqdm import trange

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2 import config
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import Visualizer

from engine.dataMapper import multiband_mapper


logger = logging.getLogger('INTELLO')


def loadDataset(cfg, split='train'):
    '''
        Currently only supports COCO-formatted datasets.
    '''
    dsName = cfg.DATASETS.NAME + '_' + split
    register_coco_instances(dsName, {},
        os.path.join(cfg.DATASETS.DATA_ROOT, split+'.json'),
        cfg.DATASETS.DATA_ROOT)
    setattr(cfg.DATASETS, split.upper(), dsName)
    if split == 'train':
        return build_detection_train_loader(cfg, dataset=DatasetCatalog.get(dsName), mapper=multiband_mapper,
                                        aspect_ratio_grouping=False)
    else:
        return build_detection_test_loader(cfg, dataset_name=dsName, mapper=multiband_mapper)



def processEpoch(cfg, epoch, dataLoader, model, train=True, resume=True):
    model.train(train)
    if train:
        optimiser = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimiser)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimiser, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get('iteration', -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    logger.info(f'[Epoch {epoch}] Starting training from iteration {start_iter}')

    # dataset loop
    tBar = trange(max_iter)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(dataLoader, range(start_iter, max_iter)):
            storage.iter = iteration

            if train:
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimiser.zero_grad()
                losses.backward()
                optimiser.step()
                storage.put_scalar('lr', optimiser.param_groups[0]['lr'], smoothing_hint=False)
                scheduler.step()
            
            else:
                print('debug')  #TODO
            
            periodic_checkpointer.step(iteration)
            tBar.update(1)

    tBar.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Detectron2 models.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/base_solarPanels.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--resume', type=int, default=1,
                        help='Whether to resume model training or start from pre-trained base.')
    args = parser.parse_args()

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    # load datasets and data loaders
    dl_train = loadDataset(cfg, 'train')
    dl_val = loadDataset(cfg, 'val')

    # initialise model
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    # adapt to multiple input bands if necessary    #TODO: dedicated routine?
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

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    # do the work
    processEpoch(cfg, 1, dl_train, model, True, True)