#!/usr/bin/env bash
set -e

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

mkdir data/coco/images
mkdir data/coco/annotations_raw

unzip -q train2017.zip -d data/coco/images
unzip -q val2017.zip -d data/coco/images
unzip -q stuffthingmaps_trainval2017.zip -d data/coco/annotations_raw