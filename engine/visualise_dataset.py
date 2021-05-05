'''
    2021 Benjamin Kellenberger
'''

import os
import argparse
import numpy as np

import matplotlib.pyplot as plt

from detectron2 import config
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

import rasterio

from engine import util


def loadDataset(cfg, split='train'):
    '''
        Currently only supports COCO-formatted datasets.
    '''
    dsName = cfg.DATASETS.NAME + '_' + split
    register_coco_instances(dsName, {},
        os.path.join(cfg.DATASETS.DATA_ROOT, split+'.json'),
        cfg.DATASETS.DATA_ROOT)
    setattr(cfg.DATASETS, split.upper(), dsName)
    return DatasetCatalog.get(dsName)


def visualise(cfg, split='train'):
    datasetDict = loadDataset(cfg, split)
    dsName = cfg.DATASETS.NAME + '_' + split

    for item in datasetDict:

        # load and visualise image
        image = util.loadImage(item['file_name'])
        
        v = Visualizer(image[:3,:,:].astype(np.uint8).transpose(1,2,0), MetadataCatalog.get(dsName), scale=1.2)
        out = v.draw_dataset_dict(item)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.title(item['file_name'])
        plt.waitforbuttonpress()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise COCO dataset.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/base_solarPanels.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--split', type=str, default='train',
                        help='Which dataset split to visualise {"train", "val", "test"}.')
    args = parser.parse_args()

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    # iterate over images
    visualise(cfg, args.split)