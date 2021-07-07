'''
    Takes the pre-train dataset (ESRI Shapefile labels and VRT) and creates COCO
    versions from it.

    2021 Benjamin Kellenberger
'''

import os
import copy
import shutil
import argparse
import tempfile
import json
from collections.abc import Iterable
from sys import meta_path
import numpy as np
from tqdm import tqdm
from osgeo import ogr, osr
ogr.UseExceptions()
osr.UseExceptions()

from engine import util
from projects.solarPanels.dataset import DataSource


def split_spatial_dataset_patches(imageSource, annotationFile, destinationFolder, patchSize, jitter=(0, 0), minArea=50, force=False):
    '''
        Similar to "split_coco_dataset_patches", but this works on GeoJSON and
        VRT sources instead of MS-COCO JSON and individual images.
    '''

    # check if already done
    if os.path.exists(destinationFolder):
        if force:
            print(f'Dataset under "{destinationFolder}" found and force recreation selected; deleting...')
            shutil.rmtree(destinationFolder)
        else:
            print(f'Dataset under "{destinationFolder}" already exists, aborting...')
            import sys
            sys.exit(0)
    
    # setup
    if not isinstance(patchSize, Iterable):
        patchSize = (patchSize, patchSize)
    halfPatch = (patchSize[0]//2, patchSize[1]//2)
    if not isinstance(jitter, Iterable):
        jitter = (jitter, jitter)
    os.makedirs(destinationFolder, exist_ok=True)

    destFile_meta = os.path.join(destinationFolder, 'train.json')

    driver = ogr.GetDriverByName('ESRI Shapefile')      # need SHP driver because GeoJSON one is buggy

    # open image source
    imgSource = DataSource(imageSource)
    resolution = imgSource.source.res
    
    # read annotation file
    ds = driver.Open(annotationFile)
    in_layer = ds.GetLayer()

    # create temporary layers for clipping
    srs =  osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    tempdir = tempfile.mkdtemp()
    ds_out = driver.CreateDataSource(os.path.join(tempdir, 'clipped.shp'))
    out_layer = ds_out.CreateLayer('polygon', srs, ogr.wkbPolygon)


    # create destination metadata
    meta_out = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': 'Solar Panel'
        }]
    }
    
    # iterate over annotations and crop patches around them
    iidx = 1        # new image index
    aidx = 1        # same for annotations
    for fIdx in tqdm(range(in_layer.GetFeatureCount())):
        feature = in_layer.GetFeature(fIdx)
        if feature.GetGeometryRef().GetGeometryType() != ogr.wkbPolygon:
            continue

        envelope = feature.GetGeometryRef().GetEnvelope()

        # skip if envelope is not valid
        if envelope[1] - envelope[0] <= 0 or envelope[3] - envelope[2] <= 0:
            continue

        centre = [(envelope[0]+envelope[1])/2, (envelope[2]+envelope[3])/2]

        # define cropping extent by jittering a bit around centre of polygon
        jitterX = 2 * (0.5 - np.random.rand(1)[0]) * jitter[0]
        jitterY = 2 * (0.5 - np.random.rand(1)[0]) * jitter[1]
        centre[0] += jitterX * resolution[1]
        centre[1] += jitterY * resolution[0]

        extent = [
            centre[0] - resolution[1] * halfPatch[0],
            centre[1] - resolution[0] * halfPatch[1],
            centre[0] + resolution[1] * (halfPatch[0]-1),
            centre[1] + resolution[0] * (halfPatch[1]-1)
        ]

        # crop and save image
        try:
            patch, patch_transform, _ = imgSource.crop(extent)
        except:
            continue
        transform_inv = ~patch_transform

        # crop, transform and save annotations
        def export_polygon(poly, image_index, annotation_index):
            poly_np = np.array(poly)
            poly_extent = [
                np.min(poly_np[:,0]),
                np.min(poly_np[:,1]),
                np.max(poly_np[:,0]),
                np.max(poly_np[:,1])
            ]
            poly_extent[0] = max(0, poly_extent[0])
            poly_extent[1] = max(0, poly_extent[1])
            poly_extent[2] = min(patchSize[0], poly_extent[2])
            poly_extent[3] = min(patchSize[1], poly_extent[3])

            poly_extent[2] -= poly_extent[0]
            poly_extent[3] -= poly_extent[1]

            if poly_extent[0] >= (patchSize[0]-1) or poly_extent[1] >= (patchSize[1]-1) or \
                poly_extent[2] <= 0 or poly_extent[3] <= 0:
                # invalid annotation; skip
                return 0

            # check and calculate area (shoelace formula)
            area = 0.5*np.abs(np.dot(poly_np[:,0],np.roll(poly_np[:,1],1))-np.dot(poly_np[:,1],np.roll(poly_np[:,0],1)))
            if area < minArea:
                return 0

            meta_out['annotations'].append({
                'id': annotation_index,
                'bbox': poly_extent,
                'area': area,
                'segmentation': [np.ravel(poly_np).tolist()],
                'category_id': 1,
                'image_id': image_index
            })
            return 1

        extent_ogr = ogr.CreateGeometryFromJson(json.dumps({'type': 'Polygon', 'coordinates': [[[extent[0], extent[1]], [extent[2], extent[1]], [extent[2], extent[3]], [extent[0], extent[3]], [extent[0], extent[1]]]]}))
        ds_clip = driver.CreateDataSource(os.path.join(tempdir, 'mask.shp'))
        clip_layer = ds_clip.CreateLayer('polygon', srs, ogr.wkbPolygon)
        clip_feat = ogr.Feature(clip_layer.GetLayerDefn())
        clip_feat.SetGeometry(extent_ogr)
        clip_layer.CreateFeature(clip_feat)

        try:
            ogr.Layer.Clip(in_layer, clip_layer, out_layer)
        except Exception as e:
            continue
        numAnno = 0
        for aIdx in range(out_layer.GetFeatureCount()):
            geom = out_layer.GetFeature(aIdx)
            poly = json.loads(geom.GetGeometryRef().ExportToJson())['coordinates']
            for p1 in range(len(poly)):
                do_segment = True
                for p2 in range(len(poly[p1])):
                    if not isinstance(poly[p1][p2][0], float):
                        # one more level
                        if poly[p1][p2][0][0] != poly[p1][p2][-1][0]:
                            poly[p1][p2].append(poly[p1][p2][-1])
                        for p3 in range(len(poly[p1][p2])):
                            poly[p1][p2][p3] = transform_inv * poly[p1][p2][p3]
                        do_segment = False
                        numAnno += export_polygon(poly[p1][p2], iidx, aidx)
                        aidx += 1
                    else:
                        poly[p1][p2] = transform_inv * poly[p1][p2]
                
                if do_segment:
                    if poly[p1][0][0] != poly[p1][-1][0]:
                        poly[p1].append(poly[p1][-1])
                    numAnno += export_polygon(poly[p1], iidx, aidx)
                    aidx += 1

        # clear gdal cache
        ds_out.DeleteLayer(0)
        out_layer = ds_out.CreateLayer('polygon', srs, ogr.wkbPolygon)

        if numAnno == 0:
            # no annotation exported for this patch; skip
            continue

        # save image
        fOut = f'{iidx}.tif'
        fileName_out = os.path.join(destinationFolder, fOut)
        util.saveImage(patch, fileName_out)

        meta_out['images'].append({
            'id': iidx,
            'file_name': fOut,
            'width': patchSize[0],
            'height': patchSize[1]
        })

        iidx += 1
        
    # save metadata file
    json.dump(meta_out, open(destFile_meta, 'w'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split GeoJSON+VRT dataset into COCO patches.')
    parser.add_argument('--image_source', type=str, default='/data/datasets/INTELLO/solarPanels_aux/images/all.vrt',
                        help='Path to the image source VRT')
    parser.add_argument('--annotation_file', type=str, default='/data/datasets/INTELLO/solarPanels_aux/labels/SolarArrayPolygons.shp',
                        help='Path to the ESRI Shapefile annotation file')
    parser.add_argument('--dest_folder', type=str, default='/data/datasets/INTELLO/solarPanels_aux/patch_datasets/224x224',
                        help='Destination directory to save patches and annotations (COCO format) into')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[224, 224],
                        help='Patch size (width, height) in pixels (default: [224, 224])')
    parser.add_argument('--jitter', type=int, nargs=2, default=[75, 75],
                        help='Random position jittering (width, height) in pixels around each annotation (default: [0, 0])')
    parser.add_argument('--min_area', type=float, default=50,
                        help='Minimum area of polygon in pixels to be kept (default: 50)')
    parser.add_argument('--force', type=int, default=1,
                        help='If 1, the dataset is forcefully being recreated (otherwise creation is skipped if image folder exists)')

    args = parser.parse_args()

    #TODO
    split_spatial_dataset_patches(args.image_source, args.annotation_file, args.dest_folder,
                                    args.patch_size, args.jitter, args.min_area, bool(args.force))