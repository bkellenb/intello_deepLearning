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
    python projects/solarPanels/dataset/create_dataset_fishnet.py --image_sources projects/solarPanels/dataset/image_sources.json \
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

    # RGB only
    python projects/solarPanels/dataset/create_dataset_fishnet.py --image_sources projects/solarPanels/dataset/image_sources_rgb.json \
                                                            --fishnet_file path/to/fishnet.shp \
                                                            --anno_file path/to/solarPanels.shp \
                                                            --anno_field Type \
                                                            --dest_folder path/to/images_rgb \
                                                            --train_frac 0.6 \
                                                            --val_frac 0.1 \
                                                            --srs EPSG:31370 \
                                                            --layers ORTHO_2020 \
                                                            --image_size 800 600 \
                                                            --image_format image/tiff;
```

Next, we merge all categories into one ("solar panel"):
```bash
  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches/train.json \
                                                            --destination_file path/to/patches/train.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches/val.json \
                                                            --destination_file path/to/patches/val.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches/test.json \
                                                            --destination_file path/to/patches/test.json;

  # RGB only
  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches_rgb/train.json \
                                                            --destination_file path/to/patches_rgb/train.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches_rgb/val.json \
                                                            --destination_file path/to/patches_rgb/val.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file path/to/patches_rgb/test.json \
                                                            --destination_file path/to/patches_rgb/test.json;
```



Finally, we split the images up into smaller patches and center five of them around
each polygon with a jitter of 25 pixels in x and y direction, and complement the
dataset by also cropping five patches at random in each image:

```bash

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/train.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/val.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images \
                                                            --annotation_file path/to/images/test.json \
                                                            --dest_folder path/to/patches \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  # RGB only
  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images_rgb \
                                                            --annotation_file path/to/images_rgb/train.json \
                                                            --dest_folder path/to/patches_rgb \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images_rgb \
                                                            --annotation_file path/to/images_rgb/val.json \
                                                            --dest_folder path/to/patches_rgb \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder path/to/images_rgb \
                                                            --annotation_file path/to/images_rgb/test.json \
                                                            --dest_folder path/to/patches_rgb \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;
```

To replicate results with slope and aspect, we can add those into a separate directory:
```bash
  python projects/solarPanels/dataset/calculate_slope_aspect.py --image_folder path/to/patches \
                                                                --dem_ordinal 4 \
                                                                --dest_folder /path/to/patches_slope_aspect;
  
  cp /path/to/patches/*.json /path/to/patches_slope_aspect/.;
```


### 2. Train models

The models can now be trained by supplying the engine with the appropriate configuration file.
For example, to train a Mask R-CNN with ResNet-50, use the following code snippet:

```bash
    python engine/train.py --config projects/solarPanels/configs/maskrcnn_r50.yaml
```


### 3. Evaluate model performance

Performance of the model on the held-out validation or test set can be evaluated
as follows:

```bash
  python engine/test.py --config projects/solarPanels/configs/maskrcnn_r50.yaml \
                        --split val \     # or "test"
                        --vis 0 \         # set to 1 to display predictions for each image
                        --evaluate 1
```

This will use the [COCO evaluation metrics](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator).


### 4. Perform inference

The code below employs a trained model and predicts polygons on a folder of
images. Images are split into patches on a regular grid according to the
`INPUT.IMAGE_SIZE` specified in the config file and processed separately.
Resulting masks are converted to polygons, transformed to match the original
image's geospatial extents and (optionally) visualised and/or exported to an
ESRI Shapefile.

```bash
  python engine/inference.py --config projects/solarPanels/configs/maskrcnn_r50.yaml \
                              --vis 1 \               # optional: set to 1 to visualise
                              --output predictions \  # optional: provide path for output Shapefile
                              --single_file 1         # set to 1 to create one SHP for all images (default)
```

Note that polygons are currently treated separately; i.e., polygons that touch each other
(e.g. for predictions that could span across inference patches) are not unified.