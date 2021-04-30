'''
    Detectron2-compliant dataset loader for the solar panels.

    2021 Benjamin Kellenberger
'''

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_solar_panels_dataset(cfg, split='train'):
    dataset = []

    #TODO: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
    annotations = []
    annotations.append({
        'bbox': [1, 2, 3, 4],
        'bbox_mode': BoxMode.XYXY_ABS,
        'category_id': 0,
        'segmentation': [
            [0.0,0.0, 1.0,0.0, 0.0,0.0]     # x1,y1, x2,y2, ..., xn,yn (abs. pixel coordinates) #TODO: DATALOADER.FILTER_EMPTY_ANNOTATIONS
        ]
    })
    dataset.append({
        'file_name': 'full_image_path',
        'width': 800,
        'height': 600,
        'image_id': 1234,
        'annotations': annotations
    })


DatasetCatalog.register('INTELLO-SolarPanels', get_solar_panels_dataset)
MetadataCatalog.get('INTELLO-SolarPanels').thing_classes = ['unknown type', 'electricity', 'warm water']