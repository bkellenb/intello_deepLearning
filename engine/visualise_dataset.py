'''
    2021 Benjamin Kellenberger
'''

import os
import argparse

import matplotlib.pyplot as plt

import torch

from detectron2 import config
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

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

    order = torch.randperm(len(datasetDict))

    for idx in order:
        item = datasetDict[idx]

        # load and visualise image
        image, extent, _ = util.loadImage(item['file_name'], 255, True)      #TODO: normalizer

        title = f'[{idx+1}/{len(datasetDict)}] ' + item['file_name'] + ' ({:.2f}, {:.2f})'.format(\
            (extent[0][0]+extent[2][0])/2, (extent[1][1]+extent[0][1])/2)
        print(title)
        
        v = Visualizer(image[:3,:,:].transpose(1,2,0), MetadataCatalog.get(dsName), scale=1.2)
        out = v.draw_dataset_dict(item)
        plt.subplot(1,2,1)
        plt.imshow(image[:3,:,:].transpose(1,2,0))
        plt.draw()
        plt.subplot(1,2,2)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.suptitle(title)
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