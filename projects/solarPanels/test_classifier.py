'''
    Testing routine for baseline classifier.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import glob
from detectron2.data.build import build_detection_test_loader
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch

from detectron2 import config
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from engine.dataMapper import MultibandMapper


from projects.solarPanels.train_classifier import load_model


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

    oa = 0.0

    for idx, data in enumerate(tqdm(dataLoader)):
        with torch.no_grad():
            pred = model(data)
            conf, yhat = pred.max(dim=1)

        if visualise:
            img_vis = (data[0]['image'][:3,:,:].permute(1,2,0) * 255).type(torch.uint8)
            label = (len(data[0]['annotations']) > 0)
            extent = data[0]['image_coords']
            loc = ((extent[0][0]+extent[2][0])/2, (extent[1][1]+extent[0][1])/2)
            title = '[conf: {:.2f}, has panel: {}]'.format(conf.item(), label) + f'[{idx+1}/{len(dataLoader)}] ' + data[0]['file_name'] + ' ({:.2f}, {:.2f})'.format(*loc)
            print(title)

            plt.imshow(img_vis)
            plt.title(title)
            plt.waitforbuttonpress()
        
        target = torch.LongTensor([(len(d['annotations']) > 0) for d in data]).to(pred.device)
        oa += torch.mean((yhat == target).float()).item()
    
    oa /= len(dataLoader)

    print('OA: {}'.format(oa))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on dataset or images; (optionally) visualise and/or perform accuracy evaluation.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/resnet50.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--split', type=str, default='test',
                        help='Which dataset split to perform inference on {"train", "val", "test"}. Ignored if "image_folder" is specified.')
    parser.add_argument('--image_folder', type=str, #default='/data/datasets/INTELLO/solarPanels/images',
                        help='Directory of images to predict on. If not specified, the dataset under "split" in the config file will be used.')
    parser.add_argument('--vis', type=int, default=1,
                        help='Whether to visualise predictions or not.')
    parser.add_argument('--evaluate', type=int, default=1,
                        help='Whether to perform accuracy evaluation or not. Ignored if "image_folder" is specified.')
    args = parser.parse_args()

    print('Initiating inference...')
    print(f'\tconfig:\t\t\t{args.config}')

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    # print quick overview of parameters
    print(f'\timage size:\t\t{cfg.INPUT.IMAGE_SIZE}')
    print(f'\tvisualise:\t\t{bool(args.vis)}')

    if args.image_folder is not None:
        load_config_dataset(cfg, split='train')     # register dataset for metadata
        dataset = load_image_folder(args.image_folder)
        print(f'\tImage folder:\t\t\t"{args.image_folder}"')
    else:
        cfg.SOLVER.IMS_PER_BATCH = 1
        if args.split != 'train':
            load_config_dataset(cfg, split='train')     # register dataset for metadata
        dataset = load_config_dataset(cfg, args.split)
        print(f'\tdataset:\t\t"{cfg.DATASETS.NAME}", split: {args.split}, no. images: {len(dataset)}')
        print(f'\tevaluate:\t\t{bool(args.evaluate)}')

    mapper = MultibandMapper(cfg.INPUT.NORMALISATION, cfg.INPUT.IMAGE_SIZE)
    dataLoader = build_detection_test_loader(dataset, mapper=mapper)
    
    # load model
    model, start_iter = load_model(cfg, resume=True)
    print(f'\tmodel iter:\t\t{start_iter}')

    print('\n')

    # do the work
    predict(cfg, dataLoader, model, args.evaluate, args.vis)