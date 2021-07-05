'''
    Merges categories togther into one according to a predefined mapping.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import json
import copy
from tqdm import tqdm

def merge_categories(annotations, output_file, mapping, remove_missing=False):
    '''
        Inputs:
            - "annotations": dict of COCO-formatted annotations or else path to COCO JSON file
            - "output_file": destination file name of the COCO-formatted annotations with merged classes
            - "mapping": dict of new name (key) and Iterable of old names (value). Example:
                        {
                            "owl": ("owl", "barred owl", "burrowing owl"),
                            "bird": ("rail", "heron")
                        }
            - "remove_missing": if True, any category not present in the mapping will be removed,
                                along with its annotations.
        
        NOTE: script does not remove any image, even if it ends up not having any annotation(s) anymore.
    '''

    # load annotations
    if isinstance(annotations, str):
        if not os.path.exists(annotations):
            raise Exception(f'Annotation file "{annotations}" could not be found.')
        meta = json.load(open(annotations, 'r'))
    else:
        meta = copy.deepcopy(annotations)

    parent, _ = os.path.split(output_file)
    os.makedirs(parent, exist_ok=True)

    # find all current categories
    categories_current = dict(zip([c['id'] for c in meta['categories']], [c for c in meta['categories']]))
    categories_current_inv = dict(zip([c['name'] for c in meta['categories']], [c['id'] for c in meta['categories']]))

    mapping_inv = {}
    for key in mapping.keys():
        if isinstance(mapping[key], str):
            mapping[key] = (mapping[key],)
        for catID in mapping[key]:
            mapping_inv[catID] = key
    
    if not remove_missing:
        # add all existing categories to the mapping if not there
        for key in categories_current.keys():
            catName = categories_current[key]['name']
            if catName not in mapping_inv:
                mapping[catName] = (catName,)
                mapping_inv[catName] = catName

    # create new indices
    mapping_ind, mapping_ind_inv = {}, {}
    categories = []
    catID = 1
    for catName in mapping.keys():
        categories.append({
            'id': catID,
            'name': catName
        })
        for catName_orig in mapping[catName]:
            catID_orig = categories_current_inv[catName_orig]
            mapping_ind[catID] = catID_orig
            mapping_ind_inv[catID_orig] = catID
        catID += 1
    
    # assign output
    meta_new = {
        'images': meta['images'],
        'annotations': [],
        'categories': categories
    }
    for key in meta.keys():
        if key not in meta_new:
            meta_new[key] = meta[key]

    for anno in tqdm(meta['annotations']):
        catID = anno['category_id']
        if catID not in mapping_ind_inv:
            continue
        anno['category_id'] = mapping_ind_inv[catID]
        meta_new['annotations'].append(anno)
    
    # save
    json.dump(meta_new, open(output_file, 'w'))

    # print new categories
    print('Re-mapping done. New categories:')
    setLen = max([len(k) for k in mapping.keys()]) + 3
    print('ID\t{}old categories'.format('new category'.ljust(setLen)))
    for cat in categories:
        catID = cat['id']
        catName = cat['name']
        old_cats = mapping[catName]
        print(str(catID) + '\t' + catName.ljust(setLen), end='')
        print(', '.join(old_cats))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Merge COCO categories together')
    parser.add_argument('--annotation_file', type=str,                      default='/data/datasets/INTELLO/solarPanels/patches_224x224/test.json',
                        help='Directory of the COCO-formatted annotation file')
    parser.add_argument('--destination_file', type=str,                            default='/data/datasets/INTELLO/solarPanels/patches_224x224_merged/test.json',
                        help='Directory to save the pruned annotation file to')
    parser.add_argument('--mapping_file', type=str,                         default='projects/solarPanels/dataset/category_map.json',
                        help='Path of the JSON-formatted category mapping file')
    parser.add_argument('--remove_missing', type=int, default=1,
                        help='Set to 1 to remove categories not present in the mapping file')
    
    args = parser.parse_args()

    mapping = json.load(open(args.mapping_file, 'r'))

    merge_categories(args.annotation_file, args.destination_file, mapping, args.remove_missing)