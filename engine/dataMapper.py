'''
    Dataset mapper that is capable of reading multi-band images.

    2021 Benjamin Kellenberger
'''

import copy
import numpy as np
from collections.abc import Iterable
import torch
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import rasterio
import rasterio.features


class MultibandMapper:
    
    def __init__(self, normalisation_value, image_size, augmentations=None):
        self.normalisation_value = normalisation_value
        
        self.transform = []
        if isinstance(augmentations, T.AugmentationList):
            self.transform.extend(augmentations.augs)
        elif isinstance(augmentations, Iterable):
            self.transform.extend(augmentations)

        self.image_size = image_size
        if self.image_size is None or (not isinstance(self.image_size, Iterable) and self.image_size < 0):
            # do not resize
            pass
        else:
            if not isinstance(self.image_size, Iterable):
                self.image_size = (self.image_size, self.image_size)
            else:
                # height, width
                self.image_size = (self.image_size[1], self.image_size[0])
            self.transform.append(T.Resize(self.image_size))
        self.transform = T.AugmentationList(self.transform)
        

    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # load multi-band image
        with rasterio.open(dataset_dict['file_name']) as f:
            image = f.read().astype(np.float32) / self.normalisation_value
            image = np.transpose(image, (1,2,0))
            coords = next(rasterio.features.shapes(f.dataset_mask(), transform=f.transform))[0]['coordinates']
        
        auginput = T.AugInput(image)
        transform = self.transform(auginput)
        image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop('annotations')
        ]
        return {
            # create the format that the model expects
            'image': image,
            'image_id': dataset_dict['image_id'],
            'image_coords': coords[0],
            'file_name': dataset_dict['file_name'],
            'annotations': annos,
            'instances': utils.annotations_to_instances(annos, image.shape[1:])
        }