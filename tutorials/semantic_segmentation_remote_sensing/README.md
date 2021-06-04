# Dense Semantic Segmentation for Remote Sensing data

Description coming soon.


## Installation

TODO.


### Prepare dataset

TODO: download data

Call function individually for each set (training, validation):

```bash
    # training
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512/train \
                                --image_products all \
                                --image_indices 1,3,5,7,9 \
                                --patch_width 512 \
                                --patch_height 512;
    
    # validation
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512/val \
                                --image_products all \
                                --image_indices 11,15,28,30,34 \
                                --patch_width 512 \
                                --patch_height 512;
```


Dataset used to train full model:

Here we are using the split proposed by Volpi et al. (2016):
```bibtex
    @article{volpi2016dense,
        title={Dense semantic labeling of subdecimeter resolution images with convolutional neural networks},
        author={Volpi, Michele and Tuia, Devis},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        volume={55},
        number={2},
        pages={881--893},
        year={2016},
        publisher={IEEE}
    }
```

```bash
    # training
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512_full/train \
                                --image_products all \
                                --image_indices 1,2,3,4,5,6,7,8,10,12,13,14,16,17,20,21,22,23,24,26,27,29,31,32,33,35,37,38 \
                                --patch_width 512 \
                                --patch_height 512;

    # validation
    python prepare_dataset.py --image_root /data/datasets/Vaihingen/images \
                                --label_root /data/datasets/Vaihingen/gts \
                                --destination /data/datasets/Vaihingen/dataset_512x512_full/val \
                                --image_products all \
                                --image_indices 11,15,28,30,34 \
                                --patch_width 512 \
                                --patch_height 512;
```