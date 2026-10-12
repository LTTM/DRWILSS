#!/usr/bin/env bash
set -e

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip

tar -xf VOCtrainval_11-May-2012.tar
unzip -q SegmentationClassAug.zip -d data/voc
unzip -q SegmentationClassAug_Visualization.zip -d data/voc
unzip -q list.zip

mv VOCdevkit/VOC2012/* data/voc
rm -rf VOCdevkit
mv list/* data/voc/splits
rm -rf list