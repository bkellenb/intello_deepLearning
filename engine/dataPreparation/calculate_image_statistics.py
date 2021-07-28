'''
    Calculates pixel mean and std values over a set of images.

    2021 Benjamin Kellenberger
'''

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
from engine.util import IMAGE_EXTENSIONS, loadImage



def _get_image_stats(imgPath):
    img, _ = loadImage(imgPath, normalisation_value=1, makeUint8=False)
    img = np.reshape(img, (img.shape[0], -1))
    means = np.mean(img, 1)
    stds = np.std(img, 1)
    return means, stds


def calc_image_stats(imageFolder, destFile, forceRecreate=False):

    if os.path.exists(destFile):
        if forceRecreate:
            print(f'File "{destFile}" found; deleting...')
            os.remove(destFile)

        else:
            print(f'File "{destFile}" found; aborting...')
            sys.exit(0)
    

    # get all images
    imgs = set()
    for ie in IMAGE_EXTENSIONS:
        imgs = imgs.union(set(glob.glob(os.path.join(imageFolder, '**/*'+ie), recursive=True)))
    imgs = tuple(imgs)

    if not len(imgs):
        print('No images found; aborting...')
        sys.exit(0)

    # prepare value vectors
    means, stds = _get_image_stats(imgs[0])
    
    for img in tqdm(imgs[1:]):
        m, s = _get_image_stats(img)
        means += m
        stds += s
    
    means /= len(imgs)
    stds /= len(imgs)

    # write out
    with open(destFile, 'w') as f:
        f.write('means:\n')
        f.write(', '.join([str(m) for m in means]) + '\n')
        f.write('stds:\n')
        f.write(', '.join([str(s) for s in stds]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate and export image statistics to text file.')
    parser.add_argument('--image_folder', type=str, default='/data/datasets/INTELLO/solarPanels_aux/patch_datasets/224x224',
                        help='Base folder for the images to calculate statistics on')
    parser.add_argument('--dest_file', type=str, default='/data/datasets/INTELLO/solarPanels_aux/patch_datasets/224x224/img_stats.txt',
                        help='Destination path for the statistics text file')
    parser.add_argument('--force_recreate', type=int, default=1,
                        help='Whether to force re-creation of splits even if files already exist (default: 0)')
    args = parser.parse_args()

    
    calc_image_stats(args.image_folder, args.dest_file, bool(args.force_recreate))