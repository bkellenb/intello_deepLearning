'''
    Receives a COCO-formatted annotations file, along with a base download URL
    for MS Azure and creates a file list that can be used with AzCopy to
    download the images of a dataset from lila.science.

    More info:
        - lila.science: http://lila.science/
        - info on how images are being accessed: http://lila.science/image-access
        - base download URLs: http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt

    2021 Benjamin Kellenberger
'''

import os
import argparse
import json


def create_lila_file_list(meta, base_url, destination_file, categories_ignore=None):

    # split base URL into URL and Azure SAS token
    baseURL, sasToken = base_url.split('?')

    if isinstance(meta, str):
        if not os.path.exists(meta):
            raise Exception(f'Annotation file "{meta}" could not be found.')
        meta = json.load(open(meta, 'r'))
    
    if categories_ignore is None:
        categories_ignore = ()
    elif isinstance(categories_ignore, str):
        categories_ignore = (categories_ignore,)
    categories_ignore = set(categories_ignore)


    imgList = []
    if not len(categories_ignore):
        # download all images (TODO: you're better off downloading the ZIP file in this case)
        for i in meta['images']:
            imgList.append(i['file_name'])
    
    else:
        # download only images from a list of categories
        categories = set([c['id'] for c in meta['categories'] if c['name'] not in categories_ignore])

        imgIDs = set()
        for a in meta['annotations']:
            if a['category_id'] in categories:
                imgIDs.add(a['image_id'])
        for i in meta['images']:
            if i['id'] in imgIDs:
                imgList.append(i['file_name'])
    
    # write to disk
    with open(destination_file, 'w') as f:
        for img in imgList:
            f.write(os.path.join(baseURL, img+sasToken) + '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Create text file of list of images to download from MS Azure (e.g., lila.science).')
    parser.add_argument('--annotation_file', type=str,
                        help='Directory of the COCO-formatted annotation file')
    parser.add_argument('--base_url', type=str,
                        help='Base URL (URL?azure_SAS_token). For lila.science, see http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt.')
    parser.add_argument('--destination_file', type=str,
                        help='Directory to save the image list file to')
    parser.add_argument('--categories_ignore', type=str, default='',
                        help='Comma-separated list of category names that should be ignored')
    
    args = parser.parse_args()


    try:
        categories_ignore = args.categories_ignore.split(',')
        categories_ignore = [c.strip() for c in categories_ignore]
    except:
        categories_ignore = []


    create_lila_file_list(args.annotation_file, args.base_url, args.destination_file, categories_ignore=None)