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
import numpy as np

from tqdm import tqdm
from osgeo import gdal
import rasterio
from rasterio import mask
import shapefile
import geojson



FIELD_NAME_VALUES = {
    'Type': {
        0: 'unknown',
        1: 'electricity',
        2: 'warm water'
    },
    'Loca': {
        0: 'on ground',
        1: 'on roof'
    }
}



def generate_coco_dataset(imageFolder, imageTilingLayer, fishnetLayer, annotationLayer, annotationField, destFolder, fractions=(0.6, 0.1, 0.3)):
    '''
        Reads an ESRI Shapefile containing fishnet polygons and assigns
        them into one of the train/val/test sets. Assignment is performed
        on a stratified random sampling basis, with each contiguous region
        being assigned to the same set to prevent spatial autocorrelation
        effects.
        Creates MS-COCO-compliant annotation files, separated into train/
        val/test splits.

        Inputs:
        - "imageFolder": base directory in which the orthoimages can be found
        - "imageTilingLayer": path to an ESRI Shapefile containing fishnet poly-
                              gons that define the tiling of the orthoimages
        - "fishnetLayer": path to an ESRI Shapefile containing fishnet polygons
        - "destFile": output path to which the split CSV file should be saved
        - "fractions": tuple of train, val, test set fractions (number of )
    '''
    print('Reading and cropping orthoimages...')
    os.makedirs(os.path.join(destFolder, 'images'), exist_ok=True)

    # create VRT from orthoimages
    vrtPath = os.path.join(imageFolder, 'orthoimages.vrt')
    if not os.path.isfile(vrtPath):
        orthoimages = glob.glob(os.path.join(imageFolder, '**/*.tif'))
        orthoVRT = gdal.BuildVRT(vrtPath, orthoimages)
        orthoVRT.FlushCache()
    with rasterio.open(vrtPath) as raster_vrt:

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
                'annotationIDs': []
            }

            # create image from perimeter
            destImagePath = 'images', f'{id}.tif'
            out_img, out_transform = mask.mask(raster_vrt, [geojson.Polygon([[
                [pCoords[0], pCoords[1]],
                [pCoords[2], pCoords[1]],
                [pCoords[2], pCoords[3]],
                [pCoords[0], pCoords[3]],
                [pCoords[0], pCoords[1]]
            ]])], crop=True)
            print('TODO: check if correct, skip if already exists') #TODO

            # update metadata
            perimeters[pCoords[0]][pCoords[1]]['file_name'] = destImagePath
            perimeters[pCoords[0]][pCoords[1]]['image_width'] = out_img.shape[2]
            perimeters[pCoords[0]][pCoords[1]]['image_height'] = out_img.shape[1]

            out_meta = raster_vrt.meta
            out_meta.update({'driver': 'GTiff',
                                'width': out_img.shape[2],
                                'height': out_img.shape[1],
                                'transform': out_transform})
            with rasterio.open(os.path.join(destFolder, destImagePath), 'w', **out_meta) as dest_img:
                dest_img.write(out_img)


    coordsX = np.array(list(perimeters.keys()))

    # load annotations
    labelClasses = {}   # dict with occurrence counts per label class
    annotations = {}
    anno = shapefile.Reader(annotationLayer)
    anno_raw = anno.shapeRecords()
    for id, anno in enumerate(anno_raw):
        label = getattr(anno.record, annotationField)
        if label not in labelClasses:
            labelClasses[label] = 0
        labelClasses[label] += 1
        annotations[id] = anno.shape.points

        # assign to perimeters: split into chunks if it exceeds borders
        for c in ((0,1), (0,3), (2,1), (2,3)):      # check all bbox corners
            coord = (anno.shape.bbox[c[0]], anno.shape.bbox[c[1]])
            matchX = coordsX[np.argmin((coordsX - coord[0])**2)]
            coordsY = np.array(list(perimeters[matchX].keys()))
            matchY = coordsY[np.argmin((coordsY - coord[1])**2)]
            perimeter = perimeters[matchX][matchY]

            if id in perimeter['annotationIDs']:
                # polygon has already been assigned to this perimeter
                continue
            perimeter = perimeter['perimeter']

            # limit and shift polygon to perimeter extents (TODO: convert to pixel coordinates)
            poly = np.array(anno.shape.points)
            poly[:,0] = np.clip(poly[:,0], perimeter[0], perimeter[2]) - perimeter[0]
            poly[:,1] = np.clip(poly[:,1], perimeter[1], perimeter[3]) - perimeter[1]

            # append
            perimeters[matchX][matchY]['annotationIDs'].append(id)
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
            
            # assign to appropriate set
            hist = np.zeros(shape=lcAbundance.shape, dtype=int)
            if len(perimeter['labels']):
                for l in perimeter['labels']:
                    hist[lcLookup[l]] += 1
            else:
                hist[-1] = 1
            
            setIdx = np.argmax(np.sum((numAnno_target - numAnno_current) - hist, 1))
            numAnno_current[setIdx,:] += hist


            # convert to MS-COCO format and append to dict
            cocoDicts[setIdx]['images'].append({
                'id': perimeter['id'],
                'file_name': perimeter['file_name'],
                'width': perimeter['image_width'],
                'height': perimeter['image_height']
            })

            for aIdx in range(len(perimeter['labels'])):
                # calculate MBR as bounding box from coordinates
                poly = perimeter['geometries'][aIdx]
                mbr = [
                    np.min(poly[:,0]),
                    np.min(poly[:,1]),
                    np.max(poly[:,0]) - np.min(poly[:,0]),
                    np.max(poly[:,1]) - np.min(poly[:,1])
                ]

                # flatten polygon into x,y-alternating pairs
                #TODO: need to remove last point?
                poly = poly.ravel()

                cocoDicts[setIdx]['annotations'].append({
                    'id': perimeter['annotationIDs'][aIdx],
                    'image_id': perimeter['id'],
                    'category_id': perimeter['labels'][aIdx],
                    'bbox': mbr,
                    'segmentation': poly,
                    # 'area': 0,
                    'iscrowd': 0
                })

            #TODO 2: write script that harvests image patches based on perimeters.



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create train/val/test split file from fishnet polygons.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/ORTHOS_2019/TIFF',
                        help='Path of the folder that contains the orthoimages')
    parser.add_argument('--image_tiling_file', type=str, default='/data/datasets/INTELLO/ORTHOS_2019/OPENDATA_ORTHOS_2019_MAILLES_TUILES_SHAPE/SPW_2019_MAILLAGE.shp',
                        help='Path to the fishnet layer (ESRI Shapefile) that defines the location of each of the orthoimages (for indexing)')
    parser.add_argument('--fishnet_file', type=str, default='datasets/solarPanels/annotations/fishnet_Wallonie_200_150m_30_4_2021_BK.shp',
                        help='Path to the fishnet layer (ESRI Shapefile) that contains perimeter polygons')
    parser.add_argument('--anno_file', type=str, default='datasets/solarPanels/annotations/SAMPLES_0_SolarPanels_30_4_2021_BK.shp',
                        help='Path to the annotation layer (ESRI Shapefile) that contains the actual label polygons')
    parser.add_argument('--anno_field', type=str, default='Type',
                        help='Name of the attribute field of the annotation layer that determines the object class')
    parser.add_argument('--dest_folder', type=str, default='TEMP_DELETE',
                        help='Destination file name for the split file to be saved into (CSV)')
    parser.add_argument('--train_frac', type=float, default=0.6,
                        help='Training set fraction (default: 0.6)')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Validation set fraction (default: 0.1)')
    args = parser.parse_args()

    fractions = [max(0, min(1, args.train_frac)), max(0, min(1, args.val_frac))]
    fractions[1] = min(fractions[1], 1 - fractions[0])
    fractions.append(max(0, 1 - sum(fractions)))

    generate_coco_dataset(args.image_folder, args.image_tiling_file,
                            args.fishnet_file, args.anno_file, args.anno_field, args.dest_folder, fractions)