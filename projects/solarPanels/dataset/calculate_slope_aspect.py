'''
    Calculate slope and aspect layers from DEM input.

    2021 Benjamin Kellenberger
'''
import os
import argparse
import glob
import numpy as np
from tqdm import tqdm

import rasterio
import richdem as rd


def calc_slope_aspect(imageFolder, demLayerOrdinal, destFolder=None):
    '''
        Receives a folder of GeoTIFFs and an ordinal denoting the layer index
        where the DEM can be found within the images. Calculates slope and
        aspect from the DEM and appends them to the images.

        If "destFolder" is None, the original images will be overwritten.
    '''
    if not imageFolder.endswith(os.sep):
        imageFolder += os.sep

    # get all images
    imgs = glob.glob(os.path.join(imageFolder, '**/*.tif'), recursive=True)
    for imgPath in tqdm(imgs):
        with rasterio.open(imgPath, 'r') as fIn:
            img = fIn.read()
            meta = fIn.meta
        
        dem = rd.rdarray(img[demLayerOrdinal,...], no_data=-1)

        # calc. slope and aspect
        slope = np.array(rd.TerrainAttribute(dem, attrib='slope_riserun'))
        aspect = np.array(rd.TerrainAttribute(dem, attrib='aspect'))

        # append to image
        img = np.concatenate((img, slope[np.newaxis,...], aspect[np.newaxis,...]), axis=0)

        # save
        if destFolder is None:
            destPath = imgPath
        else:
            destPath = os.path.join(destFolder, imgPath.replace(imageFolder, ''))
            parent, _ = os.path.split(destPath)
            os.makedirs(parent, exist_ok=True)

        meta['count'] += 2
        with rasterio.open(destPath, 'w', **meta) as f:
            f.write(img)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create and append slope and aspect from/to TIFFs.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/solarPanels/patches_224x224_5_2',
                        help='Base folder where the GeoTIFFs containing the DEM can be found.')
    parser.add_argument('--dem_ordinal', type=int, default=4,
                        help='Index (ordinal) of the layer within the GeoTIFFs where the DEM is located.')
    parser.add_argument('--dest_folder', type=str, default='/data/datasets/INTELLO/solarPanels/patches_224x224_5_2_slope_aspect',
                        help='Optional destination folder for the augmented images. If not specified, images will be overwritten in-place.')
    args = parser.parse_args()

    calc_slope_aspect(args.image_folder, args.dem_ordinal, args.dest_folder)