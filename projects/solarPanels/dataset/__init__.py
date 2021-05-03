'''
    2021 Benjamin Kellenberger
'''


#TODO
import json


ds = json.load(open('/data/datasets/INTELLO/solarPanels/val.json', 'r'))


imageIDs = set()
annoIDs = set()


for img in ds['images']:
    iid = img['id']
    if iid in imageIDs:
        raise Exception(str(iid))
    imageIDs.add(iid)

for anno in ds['annotations']:
    aid = anno['id']
    if aid in annoIDs:
        raise Exception(str(aid))
    annoIDs.add(aid)