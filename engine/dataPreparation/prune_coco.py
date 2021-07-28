'''
    Removes meta data from a COCO annotation file that fulfil at least one of
    the following conditions:
    - the image is missing
    - the image is corrupt (i.e., it cannot be loaded)
    - the image is empty (if removal of empty images is specified)
    - the annotation belongs to a category that is to be ignored (if specified)

    If categories are specified that are to be ignored, the category indices
    will be remapped.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import json
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image

from engine.util import IMAGE_EXTENSIONS


def pruneCOCOdataset(annotationFile, destinationFolder, imageRoot, categoriesIgnore=(),
                    discardMissingBBoxes=True, skipEmpty=False, skipMissing=True, convertIDsIfNeeded=True,
                    add_missing_attr=True, forceCheckCorrupt=False, forceRecreate=False):

    # setup
    destinationFile = os.path.join(destinationFolder, 'annotations_pruned.json')
    if os.path.exists(destinationFile):
        if not forceRecreate:
            print(f'Destination file "{destinationFile}" found and force recreate disabled; aborting...')
            return
        else:
            os.remove(destinationFile)

    if categoriesIgnore is None:
        categoriesIgnore = []
    elif isinstance(categoriesIgnore, str):
        categoriesIgnore = (categoriesIgnore,)
    categoriesIgnore = set(categoriesIgnore)


    # load annotation metadata
    meta = json.load(open(annotationFile, 'r'))

    # convert IDs to integers if required
    if convertIDsIfNeeded:
        # create LUT first
        lut_categories = {}
        for idx in range(len(meta['categories'])):
            catID = meta['categories'][idx]['id']
            if not isinstance(catID, int):
                meta['categories'][idx]['id'] = idx + 1
                lut_categories[catID] = idx + 1
            else:
                lut_categories[catID] = catID
        for idx in range(len(meta['annotations'])):
            meta['annotations'][idx]['category_id'] = lut_categories[meta['annotations'][idx]['category_id']]
            annoID = meta['annotations'][idx]['id']
            if not isinstance(annoID, int):
                meta['annotations'][idx]['id'] = idx + 1
    
    # add missing attributes if needed
    if add_missing_attr:
        for idx in range(len(meta['annotations'])):
            if 'iscrowd' not in meta['annotations'][idx]:
                meta['annotations'][idx]['iscrowd'] = 0
            if 'area' not in meta['annotations'][idx] and 'bbox' in meta['annotations'][idx]:
                bbox = meta['annotations'][idx]['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                meta['annotations'][idx]['area'] = area


    # filter categories
    categoryDict = dict(zip([c['id'] for c in meta['categories']], [c['name'] for c in meta['categories']]))
    categories = list(categoryDict.keys())
    categories.sort()

    categoryDict_new = {}
    categories_new = []
    categoryMap = {}
    catIndex = 1
    for cat in categories:
        if categoryDict[cat] in categoriesIgnore:
            continue
        categoryDict_new[catIndex] = categoryDict[cat]
        categories_new.append(catIndex)
        categoryMap[cat] = catIndex
        catIndex += 1

    # iterate over annotations and filter and re-map categories
    numAnno = 0
    annoDict = {}
    for a in meta['annotations']:
        if 'bbox' not in a and discardMissingBBoxes:
            continue
        if a ['category_id'] not in categoryMap:
            continue
        a['category_id'] = categoryMap[a['category_id']]

        # required for COCOEvaluator
        if 'iscrowd' not in a:
            a['iscrowd'] = 0

        if a['image_id'] not in annoDict:
            annoDict[a['image_id']] = []
        annoDict[a['image_id']].append(a)
        numAnno += 1

    # find all images on disk
    if not imageRoot.endswith(os.sep):
        imageRoot += os.sep
    
    allImages = set()
    for ext in IMAGE_EXTENSIONS:
        allImages = allImages.union(set(glob.glob(os.path.join(imageRoot, '**/*' + ext), recursive=True)))
        allImages = allImages.union(set(glob.glob(os.path.join(imageRoot, '**/*' + ext.upper()), recursive=True)))

    allImages = set([ai.replace(imageRoot, '') for ai in allImages])
    
    # corrupt images & image statistics
    doCheckCorrupt = forceCheckCorrupt
    imgs_corrupt = set()
    imgs_means, imgs_stds = np.zeros(3), np.zeros(3)
    img_count = 0
    corruptImgDefPath = os.path.join(destinationFolder, 'imgs_corrupt.txt')
    if os.path.isfile(corruptImgDefPath) and not forceCheckCorrupt:
        with open(corruptImgDefPath, 'r') as cf:
            lines = cf.readlines()
            imgs_corrupt = set([l.strip() for l in lines])
        doCheckCorrupt = False
    else:
        doCheckCorrupt = True


    # iterate over all images in metadata
    print('Checking images...')
    imgDict = {}
    for i in tqdm(meta['images']):

        fileName = i['file_name']
        if skipMissing and fileName not in allImages:
            continue
        
        if skipEmpty and i['id'] not in annoDict:
            continue

        # check integrity
        if fileName in imgs_corrupt:
            continue
        elif doCheckCorrupt:
            try:
                filePath = os.path.join(imageRoot, fileName)
                img = Image.open(filePath).convert('RGB')
                img = np.reshape(np.array(img), (-1, 3))
                imgs_means += np.mean(img, 0)
                imgs_stds += np.mean(img, 0)
                img_count += 1
            except:
                print(f'Image is corrupt: {fileName}')
                imgs_corrupt.add(fileName)
                continue

        imgDict[i['id']] = i


    if doCheckCorrupt:
        parent, _ = os.path.split(corruptImgDefPath)
        os.makedirs(parent, exist_ok=True)
        with open(corruptImgDefPath, 'w') as fc:
            for ii in imgs_corrupt:
                fc.write(ii + '\n')

        imgs_means /= img_count
        imgs_stds /= img_count
        with open(os.path.join(destinationFolder, 'imgs_stats.txt'), 'w') as fc:
            fc.write('means,' + ','.join([str(i) for i in imgs_means]) + '\n')
            fc.write('stds,' + ','.join([str(i) for i in imgs_means]) + '\n')

    print(f'Pruning complete. {len(imgDict)} images and {numAnno} annotations remain.')
    print(f'Saving to file "{destinationFile}"...')

    anno_out = []
    for val in annoDict.values():
        anno_out.extend(val)

    out = {
        'images': list(imgDict.values()),
        'annotations': anno_out,
        'categories': [{'id': c, 'name': categoryDict_new[c]} for c in categoryDict_new.keys()]
    }
    for key in meta.keys():
        if key not in out:
            out[key] = meta[key]
    
    json.dump(out, open(destinationFile, 'w'))





if __name__ == '__main__':

    parser = argparse.ArgumentParser('Prune COCO dataset')
    parser.add_argument('--annotation_file', type=str,                      default='/data/datasets/islandConservationCT/island_conservation.json',
                        help='Directory of the COCO-formatted annotation file')
    parser.add_argument('--destination_folder', type=str,                            default='/data/datasets/islandConservationCT',
                        help='Directory to save the pruned annotation file to')
    parser.add_argument('--image_folder', type=str,                            default='/data/datasets/islandConservationCT',
                        help='Directory in which the images are to be found')
    parser.add_argument('--categories_ignore', type=str, default='human',
                        help='Comma-separated list of category names that should be ignored')
    parser.add_argument('--discard_missing_bboxes', type=int, default=1,
                        help='Set to 1 to remove annotations without bounding boxes (default: 1)')
    parser.add_argument('--skip_empty', type=int, default=1,
                        help='Whether to skip images without annotations (default: 1)')
    parser.add_argument('--skip_missing', type=int, default=1,
                        help='Whether to skip images that cannot be found on disk (default: 1)')
    parser.add_argument('--convert_ids_if_needed', type=int, default=1,
                        help='Converts image, annotation and category IDs to integers if required (default: 1)')
    parser.add_argument('--add_missing_attr', type=int, default=1,
                        help='Adds attributes like "iscrowd", "area", if they are missing (default: 1)')                  
    parser.add_argument('--force_check_corrupt', type=int, default=1,
                        help='Set to 1 to force re-checking and filtering corrupt images (otherwise corrupt img names will be loaded from file). Will also recalculate mean and std. dev. values of images (default: 0)')
    parser.add_argument('--force_recreate', type=int, default=1,
                        help='Whether to force re-creation of splits even if files already exist (default: 0)')
    
    args = parser.parse_args()


    try:
        categories_ignore = args.categories_ignore.split(',')
        categories_ignore = [c.strip() for c in categories_ignore]
    except:
        categories_ignore = []


    pruneCOCOdataset(args.annotation_file, args.destination_folder, args.image_folder, categories_ignore,
                    bool(args.discard_missing_bboxes),
                    bool(args.skip_empty), bool(args.skip_missing), bool(args.convert_ids_if_needed),
                    bool(args.add_missing_attr),
                    bool(args.force_check_corrupt), bool(args.force_recreate))