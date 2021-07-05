#!/bin/bash

# Source: https://figshare.com/articles/dataset/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780
#
# 2021 Benjamin Kellenberger

dataFolder=/data/datasets/INTELLO/solarPanels_aux


mkdir -p $dataFolder


# metadata
mkdir -p "$dataFolder/labels"
wget https://ndownloader.figshare.com/articles/3385780/versions/4 -O "$dataFolder/labels/labels.zip"
unzip "$dataFolderlabels/labels.zip" -d "$dataFolder/labels"
rm -f "$dataFolder/labels/labels.zip"
ogr2ogr -f "ESRI Shapefile" "$dataFolder/labels/SolarArrayPolygons.shp" "$dataFolder/labels/SolarArrayPolygons.geojson"

# images
mkdir -p "$dataFolder/images/Stockton"
mkdir -p "$dataFolder/images/Oxnard"
mkdir -p "$dataFolder/images/Fresno"

wget https://ndownloader.figshare.com/articles/3385804/versions/1 -O "$dataFolder/images/Stockton.zip"
wget https://ndownloader.figshare.com/articles/3385807/versions/1 -O "$dataFolder/images/Oxnard.zip"
wget https://ndownloader.figshare.com/articles/3385828/versions/1 -O "$dataFolder/images/Fresno.zip"

unzip "$dataFolder/images/Stockton.zip" -d "$dataFolder/images/Stockton"
unzip "$dataFolder/images/Oxnard.zip" -d "$dataFolder/images/Oxnard"
unzip "$dataFolder/images/Fresno.zip" -d "$dataFolder/images/Fresno"

rm -f "$dataFolder/images/*.zip"


# reproject images to WGS84
for source in "$dataFolder/images/Stockton" "$dataFolder/images/Oxnard" #"$dataFolder/images/Fresno"
do 
    imgList="$(ls -1 $source/*.tif)"
    for img in $imgList
    do
        img_out="repr_$(basename -- $img)"
        gdalwarp -t_srs "+proj=longlat +ellps=WGS84" -r bilinear -of GTiff $img "$source/$img_out";
        rm $img
    done
done


# build VRT
ls -1 "$dataFolder/images/Stockton/"*.tif > "$dataFolder/images/imageList.txt"
ls -1 "$dataFolder/images/Oxnard/"*.tif > "$dataFolder/images/imageList.txt"
ls -1 "$dataFolder/images/Fresno/"*.tif > "$dataFolder/images/imageList.txt"
gdalbuildvrt "$dataFolder/images/all.vrt" -input_file_list "$dataFolder/images/imageList.txt"

# create datasets of patches
python projects/solarPanels/dataset/prepare_pretrain_dataset.py --image_source "$dataFolder/images/all.vrt" \
                                                                --annotation_file "$dataFolder/labels/SolarArrayPolygons.shp" \
                                                                --dest_folder "$dataFolder/patch_datasets/224x224" \
                                                                --patch_size 224 224;

#ln -s "$dataFolder/patch_datasets/224x224/SolarArrayPolygons.json" "$dataFolder/patch_datasets/224x224/train.json";