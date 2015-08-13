#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

# root 
DATA=/home/share/lfw/
LMDB=/home/share/shaofan/lfw_caffe/
TOOLS=$CAFFE_DIR/build/tools


# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=50
  RESIZE_WIDTH=50
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $DATA"
  exit 1
fi

echo "Destroying previous lmdb"

rm -rf $LMDB/* 

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    ./train.txt \
    $LMDB/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA \
    ./test.txt \
    $LMDB/test_lmdb

echo "Done."
