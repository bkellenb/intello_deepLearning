'''
    Script to create train/val/test splits for the solar panels data.
    Receives a vector layer containing perimeter rectangles ("fishnet")
    and assigns each perimeter rectangle to one of the three sets.
    Assignment is done on a stratified random sampling basis, with each
    contiguous region being assigned to the same set to prevent spatial
    autocorrelation effects.

    2021 Benjamin Kellenberger
'''

import os
import copy
import argparse
import glob
import shutil
import json
import numpy as np
import cv2

from tqdm import tqdm
import rasterio
import shapefile

from projects.solarPanels.dataset import FIELD_NAME_VALUES, LABEL_NAME_CONVERSIONS, DataSource, WMSSource



def generate_coco_dataset_fishnet(imageSources, fishnetLayer, annotationLayer, annotationField, destFolder, imageSize, fractions=(0.6, 0.1, 0.3), force=False):
    '''
        Reads an ESRI Shapefile containing fishnet polygons and assigns
        them into one of the train/val/test sets. Assignment is performed
        on a stratified random sampling basis, with each contiguous region
        being assigned to the same set to prevent spatial autocorrelation
        effects.
        Creates MS-COCO-compliant annotation files, separated into train/
        val/test splits.

        Inputs:
        - "imageSources": single or list of paths to TIFF image, VRT file, WMS server (dict with URL and metadata), etc.
        - "fishnetLayer": path to an ESRI Shapefile containing fishnet polygons
        - "destFolder": output path the dataset should be saved to
        - "imageSize": tuple of (width, height) for target image size
        - "fractions": tuple of train, val, test set fractions
        - "wmsMeta": dict of WMS metadata if "imageSource" is a WMS URL
    '''

    # check if already done
    imgFolderPath = os.path.join(destFolder, 'images')
    if os.path.isdir(imgFolderPath):
        if force:
            print(f'Image folder path "{imgFolderPath}" found and force recreation selected; deleting folder...')
            shutil.rmtree(imgFolderPath)
        else:
            print(f'Image folder path "{imgFolderPath}" already exists, aborting...')
            import sys
            sys.exit(0)

    if not isinstance(imageSize, tuple):
        imageSize = tuple(imageSize)

    print('Reading and cropping orthoimages...')
    os.makedirs(imgFolderPath, exist_ok=True)

    # open image sources
    imgs = []
    if isinstance(imageSources, str):
        imageSources = (imageSources,)
    for imageSource in imageSources:
        if imageSource['type'] == 'WMS':
            img = WMSSource(imageSource['url'], imageSource['wms_meta'])
        else:
            img = DataSource(imageSource['path'])
        imgs.append(img)

    
    # load annotation fishnet into custom spatial index, also clip and create images
    perimeters = {}     # dict of dicts: top left x, then y
    fishnet = shapefile.Reader(fishnetLayer)
    perimeters_raw = fishnet.shapeRecords()
    for idx, p in enumerate(tqdm(perimeters_raw)):
        # try to find ID, otherwise manually assign
        if hasattr(p.record, 'id_intello'):
            id = p.record.id_intello
        else:
            # no ID found; use index
            #FIXME: check if ID already exists
            id = idx

        pCoords = p.shape.bbox
        if not pCoords[0] in perimeters:
            perimeters[pCoords[0]] = {}
        perimeters[pCoords[0]][pCoords[1]] = {
            'id': id,
            'perimeter': np.array(p.shape.bbox),

            # assigned annotations
            'geometries': [],
            'labels': [],
            'annotationIDs': [],
            'annotationIDs_orig': set()
        }

        # create image from perimeter: compose in sources order
        destImagePath = os.path.join(destFolder, 'images', f'{id}.tif')
        
        out_img, out_transform, out_meta = None, None, None

        for sourceIdx, img in enumerate(imgs):
            img_i, img_t, img_m = img.crop(pCoords)

            img_i = np.transpose(img_i, (1,2,0)).astype(np.uint8)

            # optionally limit to specified layers (e.g., for NIR WMS we discard the red and green bands)
            layers = imageSources[sourceIdx].get('layers', None)
            if layers is not None:
                img_i = img_i[:,:,layers]

            sz = img_i.shape
            if sz[0] != imageSize[1] or sz[1] != imageSize[0]:
                # resize and adjust scaling of transform
                img_i = cv2.resize(img_i, imageSize, interpolation=cv2.INTER_CUBIC)
                img_t *= img_t.scale(sz[0]/imageSize[1], sz[1]/imageSize[0])
            
            if img_i.ndim == 2:
                img_i = img_i[...,np.newaxis]

            # append
            if out_img is None:
                out_transform = img_t
                out_meta = img_m
                out_img = img_i.astype(out_meta.get('dtype', 'float32'))
            else:
                out_img = np.concatenate((out_img, img_i), -1)

        if out_img.sum() < 1:
            print(f'WARNING: all-zero image at perimeter "{p.shape.bbox}"; skipping patch and polygon extraction for this area...')
            continue

        # transpose back
        out_img = np.transpose(out_img, (2,0,1))

        # update metadata
        perimeters[pCoords[0]][pCoords[1]]['file_name'] = destImagePath
        perimeters[pCoords[0]][pCoords[1]]['image_width'] = out_img.shape[2]
        perimeters[pCoords[0]][pCoords[1]]['image_height'] = out_img.shape[1]
        perimeters[pCoords[0]][pCoords[1]]['affine'] = out_transform

        # out_meta = raster_vrt.meta
        out_meta.update({'driver': 'GTiff',
                            'width': out_img.shape[2],
                            'height': out_img.shape[1],
                            'count': out_img.shape[0],
                            'transform': out_transform})
        with rasterio.open(destImagePath, 'w', **out_meta) as dest_img:
            dest_img.write(out_img)

    coordsX = np.array(list(perimeters.keys()))
    perimeterSize = (pCoords[2]-pCoords[0], pCoords[3]-pCoords[1])


    # load annotations
    labelClasses = {}   # dict with occurrence counts per label class
    annotations = {}
    anno = shapefile.Reader(annotationLayer)
    anno_raw = anno.shapeRecords()
    for id, anno in enumerate(anno_raw):
        label = getattr(anno.record, annotationField)
        if not isinstance(label, int):
            label = LABEL_NAME_CONVERSIONS[label]       # convert to ordinal
        label += 1   # COCO label classes start at 1
        if label not in labelClasses:
            labelClasses[label] = 0
        labelClasses[label] += 1
        annotations[id] = anno.shape.points

        # assign to perimeters: split into chunks if it exceeds borders
        for cIdx, c in enumerate(((0,1), (0,3), (2,1), (2,3))):      # check all bbox corners
            coord = (anno.shape.bbox[c[0]], anno.shape.bbox[c[1]])
            distX = (coordsX - coord[0])**2
            distX[(coordsX > coord[0]) + (coordsX+perimeterSize[0] < coord[0])] = 1e9
            matchX = coordsX[np.argmin(distX)]
            coordsY = np.array(list(perimeters[matchX].keys()))
            distY = (coordsY - coord[1])**2
            distY[(coordsY > coord[1]) + (coordsY+perimeterSize[1] < coord[1])] = 1e9
            matchY = coordsY[np.argmin(distY)]
            perimeter = perimeters[matchX][matchY]

            if id in perimeter['annotationIDs_orig']:
                # polygon has already been assigned to this perimeter
                continue

            if 'affine' not in perimeter:
                # perimeter skipped due to all-zero image
                continue

            # convert polygon to pixel coordinates (inverse affine transform)
            poly = np.array(anno.shape.points)
            for p in range(poly.shape[0]):
                poly[p,:] = ~perimeter['affine'] * (poly[p,0], poly[p,1])

            # append
            annoID = f'{id}_{cIdx}'     # need to assign new unique ID since same polygon gets split potentially multiple times
            perimeters[matchX][matchY]['annotationIDs_orig'].add(id)
            perimeters[matchX][matchY]['annotationIDs'].append(annoID)
            perimeters[matchX][matchY]['geometries'].append(poly)
            perimeters[matchX][matchY]['labels'].append(label)

    # find empty perimeters and add as separate class
    labelClasses[-1] = 0
    for pX in perimeters:
        for pY in perimeters[pX]:
            if not len(perimeters[pX][pY]['annotationIDs']):
                labelClasses[-1] += 1

    # distribute perimeters to train/val/test sets to meet expected number of annotations
    lcOrder = list(labelClasses.keys())
    lcOrder.sort()
    lcLookup = dict(zip(lcOrder, range(len(lcOrder))))
    lcAbundance = np.array([labelClasses[o] for o in lcOrder]).astype(np.float32)
    numAnno_target = np.zeros((3, len(labelClasses)), dtype=np.int32)
    numAnno_target[0,:] = np.ceil(fractions[0] * lcAbundance)
    numAnno_target[1,:] = np.ceil(fractions[1] * lcAbundance)
    numAnno_target[2,:] = np.clip(lcAbundance - numAnno_target[:2,:].sum(0), 0, None)
    numAnno_current = np.zeros_like(numAnno_target)

    # setup dicts for MS-COCO format
    cocoDict = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    for ll in lcOrder[1:]:  # skip empty class
        cocoDict['categories'].append({
            'id': ll,
            'name': FIELD_NAME_VALUES[annotationField][ll],
            'supercategory': 'solar panel'
        })
    cocoDicts = [copy.deepcopy(cocoDict) for _ in range(3)]


    # iterate in random order
    pKeysX = list(perimeters.keys())
    orderX = np.random.permutation(len(pKeysX))
    for oX in orderX:
        pKeysY = list(perimeters[pKeysX[oX]].keys())
        orderY = np.random.permutation(len(pKeysY))
        for oY in orderY:
            perimeter = perimeters[pKeysX[oX]][pKeysY[oY]]

            if 'affine' not in perimeter:
                # perimeter skipped due to all-zero image
                continue
            
            # assign to appropriate set
            hist = np.zeros(shape=lcAbundance.shape, dtype=int)
            if len(perimeter['labels']):
                for l in perimeter['labels']:
                    hist[lcLookup[l]] += 1
            else:
                hist[-1] = 1
            
            setIdx = np.argmax(np.sum((numAnno_target - numAnno_current) / numAnno_target * hist, 1))
            numAnno_current[setIdx,:] += hist


            # convert to MS-COCO format and append to dict
            cocoDicts[setIdx]['images'].append({
                'id': perimeter['id'],
                'file_name': perimeter['file_name'],
                'width': perimeter['image_width'],
                'height': perimeter['image_height']
            })

            for aIdx in range(len(perimeter['labels'])):
                
                poly = perimeter['geometries'][aIdx]

                # calculate MBR as bounding box from coordinates
                mbr = [
                    np.min(poly[:,0]),
                    np.min(poly[:,1]),
                    np.max(poly[:,0]),
                    np.max(poly[:,1])
                ]

                # check if polygon is of sufficient size
                mbr_clip = np.clip(mbr, 0, None)
                mbr_clip[2] = min(mbr_clip[2], perimeter['image_width'])
                mbr_clip[3] = min(mbr_clip[3], perimeter['image_height'])
                if mbr_clip[2]-mbr_clip[0] < 0.1 or mbr_clip[3]-mbr_clip[1] < 0.1:
                    continue
                
                # convert to XYWH format
                mbr[2] -= mbr[0]
                mbr[3] -= mbr[1]

                # area: shoelace formula
                # (https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates)
                polyArea = 0.5*np.abs(np.dot(poly[:,0],np.roll(poly[:,1],1))-np.dot(poly[:,1],np.roll(poly[:,0],1)))

                # flatten polygon into x,y-alternating pairs
                poly = poly.ravel().tolist()

                cocoDicts[setIdx]['annotations'].append({
                    'id': perimeter['annotationIDs'][aIdx],
                    'image_id': perimeter['id'],
                    'category_id': perimeter['labels'][aIdx],
                    'bbox': mbr,
                    'segmentation': [poly],
                    'area': polyArea,
                    'iscrowd': 0
                })
    
    # write annotations to files
    for setIdx, split in enumerate(('train', 'val', 'test')):
        json.dump(cocoDicts[setIdx], open(os.path.join(destFolder, split+'.json'), 'w'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create train/val/test split file from fishnet polygons.')
    parser.add_argument('--image_sources', type=str, default='projects/solarPanels/dataset/image_sources.json',
                        help='Path to a JSON file defining the image sources (directories of TIFF files, VRTs; WMS definitions, etc.)')
    parser.add_argument('--fishnet_file', type=str, default='datasets/labels_v2/fishnet_Wallonie_200_150m_12_10_2021.shp',
                        help='Path to the fishnet layer (ESRI Shapefile) that contains perimeter polygons')
    parser.add_argument('--anno_file', type=str, default='datasets/labels_v2/SAMPLES_0_SolarPanels_12_10_2021.shp',
                        help='Path to the annotation layer (ESRI Shapefile) that contains the actual label polygons')
    parser.add_argument('--anno_field', type=str, default='Classname',
                        help='Name of the attribute field of the annotation layer that determines the object class')
    parser.add_argument('--dest_folder', type=str, default='/data/datasets/INTELLO/solarPanels/v2/images_800x600',
                        help='Destination directory to save patches and annotations (COCO format) into')
    parser.add_argument('--image_size', type=int, nargs=2, default=[800, 600],
                        help='Output image size in pixels; width and height (default: [800, 600])')
    parser.add_argument('--train_frac', type=float, default=0.6,
                        help='Training set fraction (default: 0.6)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation set fraction (default: 0.1)')
    parser.add_argument('--force', type=int, default=1,
                        help='If 1, the dataset is forcefully being recreated (otherwise creation is skipped if image folder exists)')

    args = parser.parse_args()

    # parse fractions
    fractions = [max(0, min(1, args.train_frac)), max(0, min(1, args.val_frac))]
    fractions[1] = min(fractions[1], 1 - fractions[0])
    fractions.append(max(0, 1 - sum(fractions)))

    # parse image sources
    imageSources = json.load(open(args.image_sources, 'r'))['sources']

    generate_coco_dataset_fishnet(imageSources,
                            args.fishnet_file, args.anno_file, args.anno_field, args.dest_folder, args.image_size, fractions, bool(args.force))