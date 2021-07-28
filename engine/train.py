'''
    General-purpose routine for training Detectron2 models.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import logging

from tqdm import trange

import torch

import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2 import config
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.utils.events import EventStorage

from engine import util
from engine.dataMapper import MultibandMapper


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

    # data augmentation
    augmentations = []
    if hasattr(cfg, 'AUGMENTATION'):
        for aug in cfg.AUGMENTATION:
            augClass = getattr(T, aug['NAME'])
            augmentations.append(augClass(**aug['KWARGS']))

    mapper = MultibandMapper(cfg.INPUT.NORMALISATION, cfg.INPUT.IMAGE_SIZE, augmentations=augmentations)
    if split == 'train':
        return build_detection_train_loader(cfg, dataset=DatasetCatalog.get(dsName), mapper=mapper,
                                        aspect_ratio_grouping=False), \
                                        dsName
    else:
        return build_detection_test_loader(cfg, dataset_name=dsName, mapper=mapper), \
                                        dsName



def do_train(cfg, model, resume=True):

    # load dataset
    dataLoader, _ = loadDataset(cfg, 'train')
    print(f'\tdataset:\t\t"{cfg.DATASETS.NAME}", no. batches: {len(dataLoader.dataset)}')

    # load model
    model, checkpointer, start_iter = util.loadModel(cfg, resume)
    max_iter = cfg.SOLVER.MAX_ITER
    print(f'\tmodel iter:\t\t{start_iter}/{max_iter}, resume: {resume}')

    model.train()
    optimiser = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimiser)

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    print('\n')
    logger.info(f'Starting training from iteration {start_iter}')

    # dataset loop
    tBar = trange(max_iter - start_iter)
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(dataLoader, range(start_iter, max_iter)):
            storage.iter = iteration

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
            tBar.update(1)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                comm.synchronize()
            
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


    tBar.close()



def do_test(cfg, model):
    dataLoader, dsName = loadDataset(cfg, 'val')

    model.eval()

    evaluator = COCOEvaluator(dsName, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", dsName))
    results = inference_on_dataset(model, dataLoader, evaluator)
    if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dsName))
        print_csv_format(results)
    if len(results) == 1:
        results = list(results.values())[0]
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Detectron2 models.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/frcnn_r50_pretrain.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--resume', type=int, default=1,
                        help='Whether to resume model training or start from pre-trained base.')
    args = parser.parse_args()

    print('Initiating model training...')
    print(f'\tconfig:\t\t\t"{args.config}"')

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    # print quick overview of parameters
    print(f'\tbase lr:\t\t{cfg.SOLVER.BASE_LR}')
    print(f'\tscheduler:\t\t"{cfg.SOLVER.LR_SCHEDULER_NAME}", gamma: {cfg.SOLVER.GAMMA}, steps: {cfg.SOLVER.STEPS}')
    print(f'\tweight decay:\t\t{cfg.SOLVER.WEIGHT_DECAY}')
    print(f'\tbatch size:\t\t{cfg.SOLVER.IMS_PER_BATCH}')
    print(f'\tcheckpoint:\t\tsaving to "{cfg.OUTPUT_DIR}", every {cfg.SOLVER.CHECKPOINT_PERIOD} iterations')
    print(f'\timage size:\t\t{cfg.INPUT.IMAGE_SIZE}')
    if hasattr(cfg, 'AUGMENTATION'):
        print('\taugmentations:\t\t[{}]'.format(', '.join([aug['NAME'] for aug in cfg.AUGMENTATION])))
    else:
        print('\taugmentations:\t\t(None)')

    # do the work
    do_train(cfg, True)