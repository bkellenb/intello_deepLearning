#!/bin/bash

# NOTE: this requires azcopy to be installed.
#
# 2021 Benjamin Kellenberger

targetDir=/data/datasets/INTELLO/channelIslandCT
seed=23445


# do the work
mkdir -p $targetDir;

azcopy copy "https://lilablobssc.blob.core.windows.net/channel-islands-camera-traps/channel-islands-camera-traps-images.zip" \
            "$targetDir/images.zip";
unzip "$targetDir/images.zip" -d "$targetDir";
rm -f "$targetDir/images.zip";
mv "$targetDir/images/*" "$targetDir/";
rm -rf "$targetDir/images";

azcopy copy "https://lilablobssc.blob.core.windows.net/channel-islands-camera-traps/channel-islands-camera-traps.json.zip" \
            $targetDir/annotations.zip;
unzip "$targetDir/annotations.zip" -d "$targetDir";
rm -f "$targetDir/annotations.zip";


# check integrity and calculate stats
python engine/dataPreparation/prune_coco.py --annotation_file "$targetDir/channel_islands_camera_traps.json" \
                            --destination_folder "$targetDir" \
                            --image_folder "$targetDir" \
                            --categories_ignore "empty,human" \
                            --discard_missing_bboxes 1 \
                            --skip_empty 1 \
                            --skip_missing 1 \
                            --force_check_corrupt 0 \
                            --force_recreate 1;


# split into train/val/test set
python engine/dataPreparation/split_coco.py --annotation_file "$targetDir/annotations_pruned.json" \
                            --destination_folder "$targetDir" \
                            --seed $seed \
                            --percentages train 70.0 val 30.0;