'''
    2021 Benjamin Kellenberger
'''

import copy
import rasterio
from rasterio import mask, MemoryFile
import geojson
from owslib.wms import WebMapService


FIELD_NAME_VALUES = {
    'Type': {
        1: 'unknown',
        2: 'electricity',
        3: 'warm water'
    },
    'Loca': {
        1: 'on ground',
        2: 'on roof'
    }
}



class DataSource:

    def __init__(self, source):
        self.source = rasterio.open(source)
    
    def __del__(self):
        self.source.close()
    
    def crop(self, extent):
        out_img, out_transform = mask.mask(self.source, [geojson.Polygon([[
            [extent[0], extent[1]],
            [extent[2], extent[1]],
            [extent[2], extent[3]],
            [extent[0], extent[3]],
            [extent[0], extent[1]]
        ]])], crop=True)
        return out_img, out_transform, self.source.meta


class WMSSource:

    def __init__(self, wmsURL, wmsMeta):
        self.wms = WebMapService(wmsURL)

        # check WMS meta data
        for key in ('srs', 'size', 'format'):
            assert key in wmsMeta, f'Missing WMS metadata entry "{key}".'
        self.layers = wmsMeta.get('layers', None)
        if self.layers is None or not len(self.layers):
            # choose all layers
            self.layers = self.wms.contents
        elif not isinstance(self.layers, list):
            self.layers = [self.layers]
        elif isinstance(self.layers, tuple):
            self.layers = list(self.layers)
        self.srs = wmsMeta['srs']
        self.imageSize = wmsMeta['size']
        self.imageFormat = wmsMeta['format']
    
    def crop(self, extent):
        img = None
        attempt = 0
        while img is None and attempt < 5:
            if attempt > 0:
                print(f'Attempt {attempt+1}/5...')
            try:
                img = self.wms.getmap(
                    layers=self.layers,
                    bbox=tuple(extent),
                    size=self.imageSize,
                    srs=self.srs,
                    format=self.imageFormat
                )
            except Exception as e:
                img = None
        with MemoryFile(img) as memfile:
            with memfile.open() as dataset:
                out_img = dataset.read()

                # need to define custom affine matrix
                trArgs = copy.deepcopy(extent)
                trArgs.extend(self.imageSize)
                transform = rasterio.transform.from_bounds(*trArgs)

                meta = copy.deepcopy(dataset.meta)
                meta['transform'] = transform

                return out_img, transform, dataset.meta