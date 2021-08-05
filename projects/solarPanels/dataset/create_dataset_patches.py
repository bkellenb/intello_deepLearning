'''
    Script that accepts a COCO dataset and splits it into smaller patches.

    2021 Benjamin Kellenberger
'''

import os
import copy
import argparse
import json
from collections.abc import Iterable
import numpy as np
from tqdm import tqdm
from osgeo import ogr

from engine import util


def split_coco_dataset_patches(imageFolder, annotationFile, destinationFolder, patchSize, numPatchesRandom=5, numPatchesPerAnnotation=1, jitter=(0, 0), minAnnoSize=500, force=False):
    '''
        TODO
    '''
    assert imageFolder != destinationFolder, 'Image and destination folders cannot be the same.'

    # check if already done
    _, metaName = os.path.split(annotationFile)
    destFile_meta = os.path.join(destinationFolder, metaName)
    if os.path.exists(destFile_meta):
        if force:
            print(f'Annotation file "{destFile_meta}" found and force recreation selected; deleting...')
            os.remove(destFile_meta)
        else:
            print(f'Annotation file "{destFile_meta}" already exists, aborting...')
            import sys
            sys.exit(0)
    
    # setup
    if not isinstance(patchSize, Iterable):
        patchSize = (patchSize, patchSize)
    halfPatch = (patchSize[0]//2, patchSize[1]//2)
    if not isinstance(jitter, Iterable):
        jitter = (jitter, jitter)
    os.makedirs(destinationFolder, exist_ok=True)

    def crop_annotation(anno, extent):
        '''
            Receives a COCO-formatted annotation entry and crops it to the
            extent provided (tuple of left, top, right, bottom coordinates).
            Returns a zero-shifted and cropped polygon or else None if the
            annotation does not overlap with the bounding box. FIXME: currently
            only supports polygons and no pixel-wise segmentation masks.
        '''
        extent_poly = (
            (extent[0], extent[1]),
            (extent[2], extent[1]),
            (extent[2], extent[3]),
            (extent[0], extent[3]),
            (extent[0], extent[1])
        )
        extent_poly = ogr.CreateGeometryFromWkt('POLYGON (({}))'.format(
            ', '.join([f'{e[0]} {e[1]}' for e in extent_poly])
        ))
        anno_ = copy.deepcopy(anno)

        seg = anno_['segmentation']

        # close polygon if needed
        for s in range(len(seg)):
            if seg[s][0] != seg[s][-2]:
                seg[s].extend(seg[s][0:2])

        seg_poly = ogr.CreateGeometryFromJson(json.dumps({
            'type': 'Polygon',
            'coordinates': [[[subseg[s], subseg[s+1]] for s in range(0, len(subseg), 2)] for subseg in seg]
        }))

        intersect = seg_poly.Intersection(extent_poly)
        if intersect is None:
            return None

        intersect_coords = json.loads(intersect.ExportToJson())
        if 'coordinates' in intersect_coords:
            intersect_coords = intersect_coords['coordinates']
        elif 'geometries' in intersect_coords:
            intersect_coords = [g['coordinates'] for g in intersect_coords['geometries'] if g['type'] == 'Polygon']
        
        if not len(intersect_coords):
            return None

        seg_out = []
        for ii in intersect_coords:
            segment = np.array([item for sublist in ii for item in sublist])

            if segment.ndim > 1:
                segment = segment.ravel()

            # shift by patch origin
            segment[::2] -= extent[0]
            segment[1::2] -= extent[1]

            seg_out.append(segment.tolist())

        bbox_seg = list(intersect.GetEnvelope())

        anno_out = {
            'bbox': [bbox_seg[0]-extent[0], bbox_seg[2]-extent[1], bbox_seg[1]-bbox_seg[0], bbox_seg[3]-bbox_seg[2]],
            'area': intersect.GetArea(),
            'segmentation': seg_out
        }
        for key in anno.keys():
            if key not in anno_out:
                anno_out[key] = anno[key]
        return anno_out

    
    # read annotation file and create lookup dict
    meta = json.load(open(annotationFile, 'r'))
    annos = {}
    for anno in meta['annotations']:
        imgID = anno['image_id']
        if imgID not in annos:
            annos[imgID] = []
        annos[imgID].append(anno)

    # create destination metadata
    meta_out = {
        'images': [],
        'annotations': []
    }
    for key in meta.keys():
        if key not in meta_out:
            meta_out[key] = meta[key]
    
    # image and annotation indices
    imgs_existing = util.listImages(destinationFolder, True)
    iidx_offset = len(imgs_existing)       # offset image file name index to prevent referencing across sets
    iidx = 0                        # new index for image IDs (need to start at zero for COCOEvaluator)
    aidx = 0                        # new index for annotations

    # iterate over images & crop
    for ii in tqdm(meta['images']):
        
        # load image
        # img = cv2.imread(os.path.join(imageFolder, ii['file_name']), cv2.IMREAD_UNCHANGED)
        img, _ = util.loadImage(os.path.join(imageFolder, ii['file_name']), 1, False)

        annotations = annos.get(ii['id'], [])

        # crop at random
        for _ in range(numPatchesRandom):
            centre = [
                np.random.randint(halfPatch[0], img.shape[2]-halfPatch[0]),
                np.random.randint(halfPatch[1], img.shape[1]-halfPatch[1])
            ]
            extent = (
                centre[0] - halfPatch[0],
                centre[1] - halfPatch[1],
                centre[0] + halfPatch[0],
                centre[1] + halfPatch[1]
            )

            # crop and save image
            img_crop = img[:, extent[1]:extent[3], extent[0]:extent[2]]
            _, ext = os.path.splitext(ii['file_name'])
            fileName_out = os.path.join(destinationFolder, f'{iidx_offset}{ext}')
            util.saveImage(img_crop, fileName_out, {'driver': 'GTiff', 'dtype': str(img_crop.dtype)})
            # cv2.imwrite(fileName_out, img_crop)
            meta_out['images'].append({
                'id': iidx,
                'file_name': fileName_out,
                'width': img_crop.shape[2],
                'height': img_crop.shape[1]
            })

            # shift, crop and append all annotations that fall within boundaries
            for anno in annotations:
                anno_crop = crop_annotation(anno, extent)
                if anno_crop is not None:
                    anno_crop['image_id'] = iidx
                    anno_crop['id'] = aidx
                    meta_out['annotations'].append(anno_crop)
                    aidx += 1
            iidx += 1
            iidx_offset += 1
        
        # crop around annotations
        if numPatchesPerAnnotation:
            for anno in annotations:

                centre = [
                    anno['bbox'][0] + anno['bbox'][2]/2.0,
                    anno['bbox'][1] + anno['bbox'][3]/2.0
                ]

                for _ in range(numPatchesPerAnnotation):

                    # add jitter within permissible bounds
                    minJitterX = centre[0] - min(max(0, centre[0] - halfPatch[0]), jitter[0]//2)
                    minJitterY = centre[1] - min(max(0, centre[1] - halfPatch[1]), jitter[1]//2)
                    maxJitterX = centre[0] + min(patchSize[0]-1, centre[0] + halfPatch[0], jitter[0]//2)
                    maxJitterY = centre[1] + min(patchSize[1]-1, centre[1] + halfPatch[1], jitter[1]//2)
                    if minJitterX < maxJitterX:
                        centre[0] = np.random.randint(minJitterX, maxJitterX)
                    if minJitterY < maxJitterY:
                        centre[1] = np.random.randint(minJitterY, maxJitterY)
                    
                    # limit centre to image bounds
                    centre[0] = min(max(halfPatch[0], centre[0]), img.shape[2] - halfPatch[0])
                    centre[1] = min(max(halfPatch[1], centre[1]), img.shape[1] - halfPatch[1])

                    extent = (
                        centre[0] - halfPatch[0],
                        centre[1] - halfPatch[1],
                        centre[0] + halfPatch[0],
                        centre[1] + halfPatch[1]
                    )

                    # crop and save image
                    img_crop = img[:, extent[1]:extent[3], extent[0]:extent[2]]
                    _, ext = os.path.splitext(ii['file_name'])
                    fileName_out = os.path.join(destinationFolder, f'{iidx_offset}{ext}')
                    util.saveImage(img_crop, fileName_out)
                    meta_out['images'].append({
                        'id': iidx,
                        'file_name': fileName_out,
                        'width': img_crop.shape[2],
                        'height': img_crop.shape[1]
                    })

                    # shift, crop and append all annotations that fall within boundaries
                    for aa in annotations:
                        anno_crop = crop_annotation(aa, extent)
                        if anno_crop is not None and anno_crop['area'] >= minAnnoSize:
                            anno_crop['image_id'] = iidx
                            anno_crop['id'] = aidx
                            meta_out['annotations'].append(anno_crop)
                            aidx += 1
                    iidx += 1
                    iidx_offset += 1
        
    # save metadata file
    json.dump(meta_out, open(destFile_meta, 'w'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split COCO dataset into patches.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/solarPanels',
                        help='Directory in which the images are located')
    parser.add_argument('--annotation_file', type=str, default='/data/datasets/INTELLO/solarPanels/test.json',
                        help='Directory of the COCO annotation .json file')
    parser.add_argument('--dest_folder', type=str, default='/data/datasets/INTELLO/solarPanels/patches_224x224',
                        help='Destination directory to save patches and annotations (COCO format) into')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224],
                        help='Patch size (width, height) in pixels (default: [224, 224])')
    parser.add_argument('--num_patches_random', type=int, default=10,
                        help='Number of random patches per image, either with or without annotation (default: 10)')
    parser.add_argument('--num_patches_per_annotation', type=int, default=10,
                        help='Number of patches centered around each annotation, with a jitter (default: 1)')
    parser.add_argument('--min_annotation_size', type=int, default=75,
                        help='Minimum area in pixels covered by annotation (if cut) to not get discarded (default: 50)')
    parser.add_argument('--jitter', type=int, nargs=2, default=[25, 25],
                        help='Random position jittering (width, height) in pixels around each annotation (default: [0, 0])')
    parser.add_argument('--force', type=int, default=1,
                        help='If 1, the dataset is forcefully being recreated (otherwise creation is skipped if image folder exists)')

    args = parser.parse_args()

    #TODO
    split_coco_dataset_patches(args.image_folder, args.annotation_file, args.dest_folder,
                                args.patch_size, args.num_patches_random, args.num_patches_per_annotation,
                                args.jitter, args.min_annotation_size,
                                bool(args.force)
                                )