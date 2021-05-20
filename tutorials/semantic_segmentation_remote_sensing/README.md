# Dense Semantic Segmentation for Remote Sensing data

Description coming soon.


## Installation

TODO.


### Prepare dataset

TODO: download data

Call function individually for each set (training, validation, test):

```bash
    # training
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512/train \
                                --image_indices 1,3,5 \
                                --patch_width 512 \
                                --patch_height 512;
    
    # validation
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512/val \
                                --image_indices 13,15,17 \
                                --patch_width 512 \
                                --patch_height 512;
```