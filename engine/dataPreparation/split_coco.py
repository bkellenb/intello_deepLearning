'''
    Receives a COCO annotation file and splits it up into n named sets according
    to given percentages. Splits are being performed so that the categories are
    balanced according to the desired percentages.

    2021 Benjamin Kellenberger
'''

import os
import argparse
import copy
import json
import numpy as np
from tqdm import tqdm


def split_coco(annotations, output_dir, percentages, seed=-1, skip_empty=False):
    '''
        Inputs:
            - "annotations": dict of COCO-formatted annotations or else path to COCO JSON file
            - "output_dir": directory to save the output COCO JSON files to
            - "percentages": dict of set names and percentages. Example:
                                {
                                    "train": 60.0,
                                    "val": 10.0,
                                    "test": 30.0
                                }
            - "seed": int, random seed for NumPy
            - "skip_empty": set to True to ignore images without annotations
    '''
    np.random.seed(seed)
    
    # load annotations
    if isinstance(annotations, str):
        if not os.path.exists(annotations):
            raise Exception(f'Annotation file "{annotations}" could not be found.')
        meta = json.load(open(annotations, 'r'))
    else:
        meta = copy.deepcopy(annotations)
    
    os.makedirs(output_dir, exist_ok=True)

    # create LUT of images
    img_lookup = {}
    for img in meta['images']:
        img_lookup[img['id']] = img

    # parse annotations and create indices
    categories = [c['id'] for c in meta['categories']]
    numCat = len(categories)
    binEdges = list(categories) + [len(categories)]
    if not skip_empty:
        categories.append(-1)
        binEdges.append(binEdges[-1])   # ignored during calc.; just here for compatibility reasons
    categories = tuple(categories)

    category_counts = np.zeros(len(categories))
    anno_lookup = {}   # key: image ID, value: list of annotations
    for anno in meta['annotations']:
        catID = anno['category_id']
        category_counts[catID-1] += 1
        imgID = anno['image_id']
        if imgID not in anno_lookup:
            anno_lookup[imgID] = []
        anno_lookup[imgID].append(anno)
    
    # calculate expected number of annotations per category per set
    sets = tuple(percentages.keys())
    counts_expected = np.zeros((len(percentages), len(categories)))
    for idx, s in enumerate(sets):
        frac = percentages[s] / 100.0
        counts_expected[idx,:] = frac * category_counts
    counts_current = np.zeros_like(counts_expected)

    # assign in random order
    print('Assigning images to sets...')
    meta_out = {}
    for s in sets:
        meta_out[s] = {
            'images': [],
            'annotations': []
        }
        for key in meta.keys():
            if key not in meta_out[s]:
                meta_out[s][key] = meta[key]

    imgKeys = tuple(img_lookup.keys())
    order = np.random.permutation(len(img_lookup))
    for o in tqdm(order):
        imgKey = imgKeys[o]
        if imgKey not in anno_lookup:
            if skip_empty:
                continue
            else:
                labels = np.zeros(len(binEdges))
                labels[-1] = 1
                annos = None
        else:
            annos = anno_lookup[imgKey]
            labels, _ = np.histogram([a['category_id'] for a in annos], bins=binEdges)
        setIdx = np.argmax(np.sum(((counts_expected - counts_current) / counts_expected) * labels, 1))

        meta_out[sets[setIdx]]['images'].append(img_lookup[imgKey])
        if annos is not None:
            meta_out[sets[setIdx]]['annotations'].extend(annos)
        counts_current[setIdx,:] += labels
    
    # write out to files
    print('Writing out JSON files...')
    for setName in sets:
        outPath = os.path.join(output_dir, setName+'.json')
        json.dump(meta_out[setName], open(outPath, 'w'))

    # print statistics to command line
    print('Done. Statistics:')

    print(f'\tNo. categories: {numCat}')
    
    category_lookup = dict(zip([c['id'] for c in meta['categories']], [c['name'] for c in meta['categories']]))
    if not skip_empty:
        category_lookup[-1] = '(empty)'
    setLen = max(len('Category'), max([len(c) for c in category_lookup.values()])) + 1
    counts_current = counts_current.astype(np.int64)

    print('{}'.format('Category'.ljust(setLen)), end='')
    for sIdx, setName in enumerate(sets):
        print('{}'.format(setName).ljust(setLen+5), end='')
    print('total\n')
    for cat in categories:
        print(category_lookup[cat].ljust(setLen), end='')
        for sIdx, setName in enumerate(sets):
            print('{:.2f}'.format(
                counts_current[sIdx, cat-1]
            ).ljust(setLen + 5), end='')
        print('{:.2f}'.format(
            category_counts[cat-1]
        ))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Split COCO dataset into subsets')
    parser.add_argument('--annotation_file', type=str,                      default='/data/datasets/islandConservationCT/annotations_pruned.json',
                        help='Directory of the COCO-formatted annotation file')
    parser.add_argument('--destination_folder', type=str,                            default='/data/datasets/islandConservationCT',
                        help='Directory to save the pruned annotation file to')
    parser.add_argument('--skip_empty', type=int, default=1,
                        help='Set to 1 to ignore images without annotations (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for image ordering during assignment (default: 0)')
    parser.add_argument('--percentages', type=str, nargs='+', default=['train', '60.0', 'test', '40.0'],
                        help='List of name-value pairs for percentages (e.g.: "train 60.0 val 10.0 test 30.0")')
    
    args = parser.parse_args()


    # parse percentages
    perc = args.percentages
    percentages = {perc[i]: float(perc[i+1]) for i in range(0,len(perc),2)}
    perc_total = sum(percentages.values())
    if perc_total > 100.0:
        print(f'WARNING: sum of percentages exceeds 100 (value: {perc_total}); assigning in semi-random order...')


    split_coco(args.annotation_file, args.destination_folder, percentages, args.seed, bool(args.skip_empty))