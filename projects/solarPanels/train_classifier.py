'''
    Low-end baseline: train a classifier on the solar panels dataset.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import logging

from tqdm import trange

import torch

import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2 import config
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from engine.dataMapper import MultibandMapper


logger = logging.getLogger('INTELLO')





import torch.nn as nn
from torchvision.models import resnet

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.model = resnet.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data):
        imgs, labels = [], []
        for d in data:
            imgs.append(d['image'])
            labels.append(1 if len(d['annotations']) else 0)
        
        imgs = torch.stack(imgs)[:,:3,:,:].cuda()
        labels = torch.LongTensor(labels).cuda()

        if self.training:
            pred = self.model(imgs)
            loss = self.loss(pred, labels)
            return {'CE': loss}

        else:
            with torch.no_grad():
                return self.model(imgs)




def load_model(cfg, resume=True):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = Model(cfg)

    if resume:
        models = os.listdir(cfg.OUTPUT_DIR)
        if len(models):
            modelStates = [int(m.replace('.pt', '')) for m in models]
            start_iter = max(modelStates)
            stateDict = torch.load(open(os.path.join(cfg.OUTPUT_DIR, str(start_iter)+'.pt'), 'rb'), map_location='cpu')
            model.load_state_dict(stateDict['model'])
        else:
            start_iter = 0
    else:
        start_iter = 0
    model.cuda()
    return model, start_iter



def save_model(cfg, model, iter):
    stateDict = {'model': model.state_dict()}
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    torch.save(stateDict, open(os.path.join(cfg.OUTPUT_DIR, str(iter)+'.pt'), 'wb'))




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
    model, start_iter = load_model(cfg)

    max_iter = cfg.SOLVER.MAX_ITER
    print(f'\tmodel iter:\t\t{start_iter}/{max_iter}, resume: {resume}')

    model.train()
    optimiser = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimiser)

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

            # if (
            #     cfg.TEST.EVAL_PERIOD > 0
            #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            #     and iteration != max_iter - 1
            # ):
            #     do_test(cfg, model)
            #     comm.synchronize()

            if iteration > 0 and iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                save_model(cfg, model, iteration)


    tBar.close()



def do_test(cfg, model):
    raise NotImplementedError('TODO')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline classifier.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/resnet50.yaml',
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