#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12


TOOLS=$CAFFE_DIR/build/tools
DATA=/home/share/shaofan/lfw_caffe/

echo "Delete the previous mean file $DATA/lfw_mean.binaryproto"

rm -rf $DATA/lfw_mean.binaryproto

echo "Calculating the mean of training set from $DATA/train_lmdb"

$TOOLS/compute_image_mean $DATA/train_lmdb \
  $DATA/lfw_mean.binaryproto

echo "Done."
