'''
    Performs inference on original, un-cropped images in patch-wise manner and
    optionally saves them in geospatial format (if information is available).

    2021 Benjamin Kellenberger
'''

import os
import argparse
from collections.abc import Iterable
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from osgeo import ogr, osr
ogr.UseExceptions()
osr.UseExceptions()

import torch

from detectron2 import config

from imantics import Mask

from engine import util



driver = ogr.GetDriverByName('ESRI Shapefile')      # need SHP driver because the GeoJSON one is buggy
srs =  osr.SpatialReference()
srs.ImportFromEPSG(4326)


def _create_layer(outFile, layerName):
    out_layer = outFile.CreateLayer(layerName, srs, ogr.wkbPolygon)
    out_layer.CreateField(ogr.FieldDefn('img_id', ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn('img_name', ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn('conf', ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn('class_id', ogr.OFSTInt16))
    return out_layer



def predict(cfg, images, model, visualise=False, outputDir=None, outputSingleFile=True):
    model.eval()

    imgSize = cfg.INPUT.IMAGE_SIZE
    if not isinstance(imgSize, Iterable):
        imgSize = (imgSize, imgSize)

    saveOutputs = (outputDir is not None)
    if saveOutputs:
        parent, _ = os.path.split(outputDir)
        if len(parent):
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(outputDir):
            driver.DeleteDataSource(outputDir)
        out_file = driver.CreateDataSource(outputDir)
        if outputSingleFile:
            out_layer = _create_layer(out_file, 'predictions')
        else:
            os.makedirs(outputDir, exist_ok=True)

    # iterate over big images
    for idx, ii in enumerate(tqdm(images)):

        # #TODO:
        # if idx == 10:
        #     break

        img, _, transform = util.loadImage(ii['file_name'], cfg.INPUT.NORMALISATION, False)

        if saveOutputs and not outputSingleFile:
            _, imgName = os.path.split(ii['file_name'])
            imgName, _ = os.path.splitext(imgName)
            out_layer = _create_layer(out_file, imgName)

        # evaluate patch-wise
        instances = []
        geoms = {}
        sz = img.shape
        posY, posX = torch.meshgrid((torch.arange(0, sz[1], imgSize[0]), torch.arange(0, sz[2], imgSize[1])))
        for y in range(posY.size(0)):
            for x in range(posX.size(1)):
                # crop patch
                yc, xc = posY[y,x], posX[y,x]
                patch = img[:, yc:(yc+imgSize[0]), xc:(xc+imgSize[1])]
                patch = torch.from_numpy(patch)

                # get prediction
                pred = model([{'image': patch}])
                pred = pred[0]['instances']
                if len(pred):

                    for pr in range(len(pred)):
                        
                        #TODO: skip step if no masks (e.g., Faster R-CNN) and export bboxes instead
                        mask = pred.pred_masks[pr,...]
                        label = pred.pred_classes[pr].item()
                        if label not in geoms:
                            geoms[label] = []
                        polygons = Mask(mask.cpu().numpy()).polygons()
                        for poly in polygons.points:
                            # offset w.r.t. patch/image origin
                            poly = poly.astype(np.float32)
                            poly[:,0] += yc.item()
                            poly[:,1] += xc.item()

                            # append to image-wide list for visualisation
                            instances.append(poly)

                            # append to output layer
                            if saveOutputs:
                                # geocode polygon
                                poly_spatial = poly.copy()
                                for p in range(poly_spatial.shape[0]):
                                    poly_spatial[p,:] = transform * poly_spatial[p,:]

                                geom = ogr.CreateGeometryFromJson(
                                    json.dumps({
                                        'type': 'Polygon',
                                        'coordinates': [poly_spatial.tolist()]
                                    })
                                )
                                # geom.CloseRings()
                                # try:
                                #     geom = geom.MakeValid()
                                # except:
                                #     print('debug')
                                geoms[label].append(geom)
                                feature = ogr.Feature(out_layer.GetLayerDefn())
                                feature.SetGeometry(geom)
                                feature.SetField('img_id', ii['image_id'])
                                feature.SetField('img_name', ii['file_name'])
                                feature.SetField('class_id', label)
                                feature.SetField('conf', pred.scores[pr].item())
                                try:
                                    out_layer.CreateFeature(feature)
                                except:
                                    print('debug')

        if visualise:
            img_vis, _, _ = util.loadImage(ii['file_name'], 1, False)
            img_vis = img_vis[:3,...].transpose(1,2,0).astype(np.uint8)
            plt.figure(1)
            plt.clf()
            plt.imshow(img_vis)
            ax = plt.gca()
            for i in instances:
                poly = Polygon(i, fc=(0,0,1,0.5), ec=(0,0,1,1), lw=0.1)
                ax.add_patch(poly)
            plt.title(ii['file_name'])
            plt.waitforbuttonpress()


        # # calculate union of polygons across patches      #TODO: buggy due to corrupt polygons
        # if saveOutputs:
        #     for label in geoms:
        #         geom_out = ogr.Geometry(ogr.wkbMultiPolygon)
        #         for gg in geoms[label]:
        #             geom_out.AddGeometryDirectly(gg)
        #         feature = ogr.Feature(out_layer.GetLayerDefn())
        #         try:
        #             feature.SetGeometry(geom_out.UnionCascaded())
        #         except:
        #             print('debug')
        #         feature.SetField('img_id', ii['image_id'])
        #         feature.SetField('img_name', ii['file_name'])
        #         feature.SetField('class_id', label)
        #         out_layer.CreateFeature(feature)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on large images, optionally visualise and/or save in geospatial format.')
    parser.add_argument('--config', type=str, default='projects/solarPanels/configs/maskrcnn_r50_slopeAspect.yaml',
                        help='Path to the config.yaml file to use on this machine.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/solarPanels/images_slope_aspect',
                        help='Directory of images to predict on.')
    parser.add_argument('--vis', type=int, default=1,
                        help='Whether to visualise predictions or not.')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Destination to save predictions to. Note that this may be interpreted as a folder or a file depending on the model.')
    parser.add_argument('--single_file', type=int, default=1,
                        help='Set to 1 to save predictions to a single file (default); 0 creates one file per image.')
    parser.add_argument('--start_iter', type=int, default=-1,
                        help='Starting iteration for model to load (default: -1 for latest)')
    args = parser.parse_args()

    print('Initiating inference...')
    print(f'\tconfig:\t\t\t{args.config}')

    # load config
    cfg = config.get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)

    # load images
    imgs = util.load_image_folder(args.image_folder)

    # load model
    model, _, start_iter = util.loadModel(cfg, resume=True, startIter=args.start_iter)

    # print quick overview of parameters
    print(f'\timage folder:\t\t{args.image_folder}')
    print(f'\t# images:\t\t{len(imgs)}')
    print(f'\tpatch size:\t\t{cfg.INPUT.IMAGE_SIZE}')
    print(f'\tvisualise:\t\t{bool(args.vis)}')
    print(f'\tsave to:\t\t{args.output}')
    print(f'\tmodel iter:\t\t{start_iter}\n')

    # do the work
    predict(cfg, imgs, model, bool(args.vis), args.output, bool(args.single_file))