'''
    Dataset mapper that is capable of reading multi-band images.

    2021 Benjamin Kellenberger
'''

import copy
import numpy as np
import torch
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import rasterio


def multiband_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # load multi-band image
    with rasterio.open(dataset_dict['file_name']) as f:
        image = f.read().astype(np.float32) / 65535 # * 255     #TODO: dirty image normalisation
        image = np.transpose(image, (1,2,0))

    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((800, 800))(auginput)  #TODO
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
       # create the format that the model expects
       "image": image,
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }