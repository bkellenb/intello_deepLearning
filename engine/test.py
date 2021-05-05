'''
    Testing routines that include prediction, visualisation
    and/or accuracy evaluation.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import glob
from detectron2.data.build import build_detection_test_loader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from detectron2 import config
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

from engine import util
from engine.dataMapper import MultibandMapper


IMAGE_EXTENSIONS = (
    '.tif',
    '.tiff',
    '.jpg',
    '.jpeg',
    '.png'
)



def load_config_dataset(cfg, split='train'):
    '''
        Currently only supports COCO-formatted datasets.
    '''
    dsName = cfg.DATASETS.NAME + '_' + split
    register_coco_instances(dsName, {},
        os.path.join(cfg.DATASETS.DATA_ROOT, split+'.json'),
        cfg.DATASETS.DATA_ROOT)
    setattr(cfg.DATASETS, split.upper(), dsName)
    return DatasetCatalog.get(dsName)



def load_image_folder(directory):
    images = set()
    for ie in IMAGE_EXTENSIONS:
        images = images.union(set(glob.glob(os.path.join(directory, '**/*'+ie), recursive=True)))
        images = images.union(set(glob.glob(os.path.join(directory, '**/*'+ie.upper()))))
    
    # bring into minimal Detectron2-compliant form
    images = [{'image_id': idx, 'file_name': i, 'annotations': []} for idx, i in enumerate(images)]
    return images



def predict(cfg, dataLoader, model, evaluate=False, visualise=False):
    model.eval()

    dsName = cfg.DATASETS.NAME+'_train'

    for idx, data in enumerate(tqdm(dataLoader)):
        with torch.no_grad():
            pred = model(data)

        if visualise:
            img_vis = (data[0]['image'][:3,:,:].permute(1,2,0) * 255).type(torch.uint8)
            v = Visualizer(img_vis, MetadataCatalog.get(dsName), scale=1.2)
            out = v.draw_instance_predictions(pred[0]['instances'].to('cpu'))
            plt.imshow(out.get_image()[:, :, ::-1])
            plt.title(data[0]['file_name'])
            plt.waitforbuttonpress()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on dataset or images; (optionally) visualise and/or perform accuracy evaluation.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/base_solarPanels.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--split', type=str, default='test',
                        help='Which dataset split to perform inference on {"train", "val", "test"}. Ignored if "image_folder" is specified.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/solarPanels/images',
                        help='Directory of images to predict on. If not specified, the dataset under "split" in the config file will be used.')
    parser.add_argument('--vis', type=int, default=1,
                        help='Whether to visualise predictions or not.')
    parser.add_argument('--evaluate', type=int, default=1,
                        help='Whether to perform accuracy evaluation or not. Ignored if "image_folder" is specified.')
    args = parser.parse_args()

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    load_config_dataset(cfg, split='train')     # register dataset for metadata
    if args.image_folder is not None:
        dataset = load_image_folder(args.image_folder)
    else:
        cfg.SOLVER.IMS_PER_BATCH = 1
        dataset = load_config_dataset(cfg, args.split)

    mapper = MultibandMapper(cfg.INPUT.NORMALISATION, cfg.INPUT.IMAGE_SIZE)
    dataLoader = build_detection_test_loader(dataset, mapper=mapper)
    
    # load model
    model, _, start_iter = util.loadModel(cfg)
    print(f'Loaded model state at iteration {start_iter}.')

    # do the work
    predict(cfg, dataLoader, model, args.evaluate, args.vis)