'''
    Auxiliary script that prepares a subset of the Vaihingen dataset for the
    tutorial.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import glob
import re
import numpy as np

from tqdm import tqdm
from PIL import Image


INDEX_PATTERN = re.compile('[0-9]+\..+$')

IMG_EXTENSIONS = (
    '.png',
    '.tif',
    '.jpg'
)

VAIHINGEN_COLORMAP = (
    (255, 255, 255),   # 1  Impervious surfaces
    (0, 0, 255),       # 2  Buildings 
    (0, 255, 255),     # 3  Low vegetation
    (0, 255, 0),       # 4  Tree 
    (255, 255, 0),     # 5  Car
    (255, 0, 0)        # 6  Clutter / background
)


def _get_images(directory):
    '''
        Recursively scans a folder and retrieves all image file names.
    '''
    imgs= set()
    for ie in IMG_EXTENSIONS:
        imgs = imgs.union(set(glob.glob(os.path.join(directory, '**/*' + ie), recursive=True)))
    return imgs



def _extract_index(fileName):
    '''
        Extracts the numerical index from the file name, which, in the Vaihingen
        dataset, is the last number before the file extension dot.
    '''
    index = INDEX_PATTERN.findall(fileName)
    if len(index):
        index = index[-1]
        return index[:index.rfind('.')]
    else:
        return None



def prepare_vaihingen_dataset(image_root, label_root, destination, image_indices='all', patchSize=(512, 512)):
    '''
        Receives a directory of images (with subdirectories for each image type,
        if needed), labels and a destination directory. Then, the function
        retrieves those image-label pairs whose indices are in "image_indices"
        (unless the latter is specified as 'all', in which case all images are
        being retrieved). Those images are then split into patches of size
        "patchSize", and the result is saved into the "destination".
    '''

    if not image_root.endswith(os.sep):
        image_root += os.sep

    # find all semantic segmentation patches
    labelFiles = _get_images(label_root)

    if image_indices in ('all', ['all']):
        # the segmentation patches define which ones to keep
        image_indices = set()
        for lf in labelFiles:
            index = _extract_index(lf)
            if index is not None:
                image_indices.add(index)
    else:
        image_indices = set(image_indices)

    # create a dict of image-label pairs
    imgLabelPairs = dict.fromkeys(image_indices)

    # iterate over label patches
    for labelFile in labelFiles:
        index = _extract_index(labelFile)
        if index in image_indices:
            imgLabelPairs[index] = {
                'images': {},
                'labels': labelFile
            }
    
    # iterate over image patches and find all image types (patch, DSM, nDSM, etc.)
    imageFiles = _get_images(image_root)
    imageTypes = set()
    for imgFile in imageFiles:
        index = _extract_index(imgFile)
        if index in imgLabelPairs:
            imageType = os.path.split(imgFile)[0].replace(image_root, '')   # image type: name of parent folder
            imageTypes.add(imageType)
            imgLabelPairs[index]['images'][imageType] = imgFile

    # now that we have all images, let's split them up into patches
    imageTypes = tuple(imageTypes)
    indexList = ['label_file,{}'.format(
        ','.join(imageTypes)
    )]          # list of image-label pairs for our CSV file
    for index in tqdm(imgLabelPairs):

        nextImageSet = imgLabelPairs[index]

        dest_images = f'{destination}/images'
        os.makedirs(dest_images, exist_ok=True)

        dest_labels = f'{destination}/labels'
        os.makedirs(f'{dest_labels}/{index}', exist_ok=True)

        # load images and labels
        imgFiles = nextImageSet['images']
        imgs = [Image.open(imgFiles[imgType]) for imgType in imageTypes]
        imgNames = [os.path.split(imgFiles[imgType])[0].replace(image_root, '') for imgType in imageTypes]        # keep parent directory name...
        imgExt = [os.path.splitext(imgFiles[imgType])[1] for imgType in imageTypes]                               # ...and image extension
        labels_raw = np.array(Image.open(nextImageSet['labels']), dtype=np.uint8)
        labels_shape = labels_raw.shape
        labels_raw = np.reshape(labels_raw, (-1, 3))

        # convert colors to label indices
        labels = 255 * np.ones(np.prod(labels_shape[:2]), dtype=np.uint8)       # 255 = unlabeled
        for lIdx, color in enumerate(VAIHINGEN_COLORMAP):
            valid = (np.sum(labels_raw - color, 1) == 0)
            labels[valid] = lIdx
        labels = Image.fromarray(np.reshape(labels, labels_shape[:2]))

        # split into patches and export each one
        for x in range(0, labels.width, patchSize[0]):
            for y in range(0, labels.height, patchSize[1]):

                # crop and save images...
                indexNames = []
                for i in range(len(imgs)):
                    os.makedirs(f'{dest_images}/{index}/{imgNames[i]}', exist_ok=True)
                    indexName = f'{index}/{imgNames[i]}/{x}_{y}{imgExt[i]}'
                    indexNames.append(indexName)
                    imgName = f'{dest_images}/{indexName}'
                    crop = imgs[i].crop((x, y, (x+patchSize[0]), (y+patchSize[1])))
                    crop.save(imgName)

                # ...and labels
                labelIndexName = f'{index}/{x}_{y}{imgExt[i]}'
                labelName = f'{dest_labels}/{labelIndexName}'
                crop = labels.crop((x, y, (x+patchSize[0]), (y+patchSize[1])))
                crop.save(labelName)

                indexList.append('{},{}'.format(
                    labelIndexName,
                    ','.join(indexNames)
                ))

    # finally, save CSV file of image names
    with open(os.path.join(destination, 'fileList.csv'), 'w') as f:
        for i in indexList:
            f.write(i + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Take subset of Vaihingen dataset and split it into patches.')
    parser.add_argument('--image_root', type=str,               default = '/data/datasets/Vaihingen/images',
                        help='Base directory where all the images are contained in (excl. label files)')
    parser.add_argument('--label_root', type=str,               default = '/data/datasets/Vaihingen/gts',
                        help='Base directory where all the label files are contained in')
    parser.add_argument('--destination', type=str,              default = '/data/datasets/Vaihingen/dataset_512x512',
                        help='Base output directory to save the patch-based dataset to')
    parser.add_argument('--image_indices', type=str, default='all',
                        help='Comma-separated list of integers of image indices to restrict dataset to (default: "all" for no restriction)')
    parser.add_argument('--patch_width', type=int, default=512,
                        help='Width in pixels of the patches to extract (default: 512)')
    parser.add_argument('--patch_height', type=int, default=512,
                        help='Height in pixels of the patches to extract (default: 512)')
    args = parser.parse_args()


    imageIndices = (args.image_indices.split(',') if hasattr(args, 'image_indices') else [])

    prepare_vaihingen_dataset(
        args.image_root,
        args.label_root,
        args.destination,
        imageIndices,
        (args.patch_width, args.patch_height)
    )