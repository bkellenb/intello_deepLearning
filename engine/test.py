'''
    Testing routines that include prediction, visualisation
    and/or accuracy evaluation.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import glob
from tqdm import trange
from detectron2.data.build import build_detection_test_loader
import matplotlib.pyplot as plt

import torch

from detectron2 import config
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, CityscapesInstanceEvaluator
from detectron2.utils.visualizer import Visualizer, ColorMode

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

    dataset = DatasetCatalog.get(dsName)
    return dataset



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

    dsName = cfg.DATASETS.NAME+'_train'       # for categories

    if evaluate:
        evaluator = COCOEvaluator(dsName, use_fast_impl=False)
        evaluator.reset()

    tBar = trange(len(dataLoader))
    for idx, data in enumerate(dataLoader):
        with torch.no_grad():
            pred = model(data)

        tBar.set_description_str('[{}] # pred: {}; # target: {}'.format(
            data[0]['file_name'],
            len(pred[0]['instances']),
            len(data[0]['instances'])
        ))
        tBar.update(1)

        if evaluate:
            evaluator.process(data, pred)

        if visualise:
            img_vis = (data[0]['image'][:3,:,:].permute(1,2,0) * cfg.INPUT.NORMALISATION).type(torch.uint8)
            v_gt = Visualizer(img_vis, MetadataCatalog.get(dsName), scale=2.0, instance_mode=ColorMode.IMAGE_BW)
            out_gt = v_gt.draw_dataset_dict(data[0])
            v_pred = Visualizer(img_vis, MetadataCatalog.get(dsName), scale=2.0, instance_mode=ColorMode.IMAGE_BW)
            out_pred = v_pred.draw_instance_predictions(pred[0]['instances'].to('cpu'))

            extent = data[0]['image_coords']
            loc = ((extent[0][0]+extent[2][0])/2, (extent[1][1]+extent[0][1])/2)
            title = f'[{idx+1}/{len(dataLoader)}] ' + data[0]['file_name'] + ' ({:.2f}, {:.2f})'.format(*loc)

            plt.subplot(1,2,1)
            plt.imshow(out_gt.get_image()[:, :, ::-1])
            plt.title('ground truth')

            plt.subplot(1,2,2)
            plt.imshow(out_pred.get_image()[:, :, ::-1])
            plt.title('prediction')
            plt.suptitle(title)
            plt.waitforbuttonpress()

    if evaluate:
        accuracyMetrics = evaluator.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on dataset or images; (optionally) visualise and/or perform accuracy evaluation.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/maskrcnn_r50.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--split', type=str, default='val',
                        help='Which dataset split to perform inference on {"train", "val", "test"}. Ignored if "image_folder" is specified.')
    parser.add_argument('--image_folder', type=str, #default='/data/datasets/INTELLO/solarPanels/images',
                        help='Directory of images to predict on. If not specified, the dataset under "split" in the config file will be used.')
    parser.add_argument('--vis', type=int, default=0,
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
    evaluate = bool(args.evaluate)

    # print quick overview of parameters
    print(f'\timage size:\t\t{cfg.INPUT.IMAGE_SIZE}')
    print(f'\tvisualise:\t\t{bool(args.vis)}')

    if args.image_folder is not None:
        load_config_dataset(cfg, split='train')     # register dataset for metadata
        dataset = load_image_folder(args.image_folder)
        evaluate = False        # cannot evaluate without ground truth
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
    model, _, start_iter = util.loadModel(cfg, resume=True)
    print(f'\tmodel iter:\t\t{start_iter}')

    print('\n')

    # do the work
    predict(cfg, dataLoader, model, evaluate, args.vis)