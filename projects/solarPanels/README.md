# Solar panel delineation

This project employs deep learning models to delineate solar panels in aerial imagery.
Currently supported tasks and models:
* Instance segmentation: Mask R-CNN
* Detection: Faster R-CNN
* Semantic segmentation: U-Net _(beta)_


## Data sources

Experiments have been conducted using the following types of remote sensing imagery:
  * RGB orthoimages from 2020: `https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_2020/MapServer/WMSServer`
  * Digital Height Model (DHM; `MNH_ORTHOS_2019.tif`)
  * Digital Surface Model (DSM; `MNS2019.tif`)
  * Slope and aspect (calculated separately from DHM)


## Current results

Statistical figures below are calculated on held-out validation set using Detectron2's COCOEvaluator.

### Mask R-CNN (RGB+DHM+DSM)

_(Iteration 500'000)_

`python engine/test.py --config projects/solarPanels/configs/maskrcnn_r50.yaml`

**BBOX**
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.065 |
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.157 |
| Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.041 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.145 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.038 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.054 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.109 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.250 |

**SEGM**
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.046 |
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.135 |
| Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.019 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.025 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.184 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.043 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.082 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.082 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.251 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |



**Mask R-CNN (RGB+DHM+DSM+slope+aspect)**

_(Iteration 500'000)_

`python engine/test.py --config projects/solarPanels/configs/maskrcnn_r50_slopeAspect.yaml`

**BBOX**
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.043 |
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.108 |
| Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.019 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.157 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.030 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.035 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.078 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.078 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.228 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.212 |

**SEGM**
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.033 |
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.085 |
| Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.015 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.150 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.030 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.054 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.054 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.211 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |



### Faster R-CNN (RGB+DHM+DSM)

_(Iteration 500'000)_

`python engine/test.py --config projects/solarPanels/configs/frcnn_r50.yaml`

**BBOX**
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.020 |
| Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.059 |
| Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.011 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.097 |
| Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.022 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.078 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.080 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.225 |
| Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000 |



### U-Net (RGB+DHM+DSM)

_(Iteration 500'000)_

`python engine/test.py --config projects/solarPanels/configs/unet.yaml`

| Precision | 0.296 |
| Recall | 0.258 |



### U-Net (RGB+DHM+DSM+slope+aspect)

_(Iteration 170'000, trained and validated on images of size 800x600)_

`python engine/test.py --config projects/solarPanels/configs/unet_slopeAspect_800x600.yaml`

| Precision | 0.338 |
| Recall | 0.404 |



## Results replication

To replicate the results obtained in here, please follow the steps as described below.


### 1. Dataset preparation

This part depends on three main inputs:
  * A fishnet shapefile defining the extents that were annotated for solar panels
  * A polygon shapefile containing solar panel polygons themselves
  * A JSON file containing the definition of input layers in order. See file [dataset/image_sources.json](dataset/image_sources.json) as an example for RGB+DHM+DSM.

These inputs are then used to create a dataset of images (default size 800x600) along with annotations in [MS-COCO format](https://cocodataset.org) containing category labels, bounding boxes, and polygons (in image coordinates):

```bash
    # parameters
    fishnetPath=path/to/fishnet.shp               # location of the Shapefile containing fishnet polygons that were annotated by the INTELLO team
    solarPanelsPath=path/to/solarPanels.shp       # location of the Shapefile containing the actual solar panel polygon annotations
    destFolder_800x600=images                     # destination directory for the images cropped with the fishnet polygon extents
    destFolder_224x224=patches                    # destination directory for the smaller patches cropped from the images for model training


    python projects/solarPanels/dataset/create_dataset_fishnet.py --image_sources projects/solarPanels/dataset/image_sources.json \
                                                            --fishnet_file $fishnetPath \
                                                            --anno_file $solarPanelsPath \
                                                            --anno_field Type \
                                                            --dest_folder $destFolder_800x600 \
                                                            --train_frac 0.6 \
                                                            --val_frac 0.1 \
                                                            --srs EPSG:31370 \
                                                            --layers ORTHO_2020 \
                                                            --image_size 800 600 \
                                                            --image_format image/tiff;
```

Next, we merge all categories ("warm water", "electricity", "unknown") into one ("solar panel"):
```bash
  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file $destFolder_800x600/train.json \
                                                            --destination_file $destFolder_800x600/train.json \
                                                            --mapping_file projects/solarPanels/dataset/category_map.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file $destFolder_800x600/val.json \
                                                            --destination_file $destFolder_800x600/val.json \
                                                            --mapping_file projects/solarPanels/dataset/category_map.json;

  python projects/solarPanels/dataset/coco_merge_categories.py --annotation_file $destFolder_800x600/test.json \
                                                            --destination_file $destFolder_800x600/test.json \
                                                            --mapping_file projects/solarPanels/dataset/category_map.json;
```



Finally, we split the images up into smaller patches and center five of them around
each polygon with a jitter of 25 pixels in x and y direction, and complement the
dataset by also cropping five patches at random in each image:

```bash

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder $destFolder_800x600 \
                                                            --annotation_file $destFolder_800x600/train.json \
                                                            --dest_folder $destFolder_224x224 \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder $destFolder_800x600 \
                                                            --annotation_file $destFolder_800x600/val.json \
                                                            --dest_folder $destFolder_224x224 \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;

  python projects/solarPanels/dataset/create_dataset_patches.py --image_folder $destFolder_800x600 \
                                                            --annotation_file $destFolder_800x600/test.json \
                                                            --dest_folder $destFolder_224x224 \
                                                            --patch_size 224 224 \
                                                            --num_patches_random 5 \
                                                            --num_patches_per_annotation 5 \
                                                            --jitter 25 25;
```

To replicate results with slope and aspect, we can add those into a separate directory:
```bash
  python projects/solarPanels/dataset/calculate_slope_aspect.py --image_folder $destFolder_224x224 \
                                                                --dem_ordinal 4 \       # band index of the images where the DEM is to be found (starts at zero)
                                                                --dest_folder ${destFolder_224x224}_slope_aspect;
```
If a separate directory is specified (parameter `--dest_folder`), any annotation metadata files (*.json) will be copied and modified as well.


Finally, you may need to modify the source paths in the [configuration scripts](configs), files `base_*.yaml`:
```yaml
# ...
DATASETS:
  DATA_ROOT: "patches"
# ...
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