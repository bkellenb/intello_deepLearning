# Solar panel delineation

This project employs deep learning models to delineate solar panels in aerial imagery.
Candidate tasks and models:
* __(current favourite)__ Instance segmentation: Mask R-CNN
* Semantic segmentation: DeepLabV3+
* Detection: Faster R-CNN



## Results replication

To replicate the results obtained in here, please follow the steps as described below.


### 1. Dataset preparation

This part depends on three main inputs:
  * A fishnet shapefile defining the extents that were annotated for solar panels
  * A polygon shapefile containing solar panel polygons themselves
  * A path or an URL pointing to a GeoTIFF image, a folder of GeoTIFF images, a
    VRT file, or else a Web Map Service (WMS) under which raster data can be retrieved. By default we use the Wallonian orthoimages from 2020 as a source: `https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_2020/MapServer/WMSServer`.

These inputs are then used to create a dataset of images (default size 800x600) along with annotations in [MS-COCO format](https://cocodataset.org) containing category labels, bounding boxes, and polygons (in image coordinates):

```bash

    python projects/solarPanels/dataset/create_dataset_fishnet.py --image_source https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_2020/MapServer/WMSServer \
                                                            --fishnet_file path/to/fishnet.shp \
                                                            --anno_file path/to/solarPanels.shp \
                                                            --anno_field Type \
                                                            --dest_folder path/to/images \
                                                            --train_frac 0.6 \
                                                            --val_frac 0.1 \
                                                            --srs EPSG:31370 \
                                                            --layers ORTHO_2020 \
                                                            --image_size 800 600 \
                                                            --image_format image/tiff;
```

Next, we split the images up into smaller patches and center five of them around
each polygon with a jitter of 25 pixels in x and y direction, and complement the
dataset by also cropping five patches at random in each image:

```bash

  python projects/solarPanels/dataset/create_dataset_fishnet.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/train.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_fishnet.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/val.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_fishnet.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/test.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;
```



### 2. Train models

The models can now be trained by supplying the engine with the appropriate configuration file.
For example, to train a Mask R-CNN with ResNet-50, use the following code snippet:

```bash
    python engine/train.py --config projects/solarPanels/configs/maskrcnn_r50.yaml
```


### 3. Evaluate model performance

Coming soon.