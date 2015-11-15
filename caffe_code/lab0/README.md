LFW - caffe
-----------

- [abstract\_name.py]: create a picture list
    - train.txt
    - test.txt
- [create\_dataset.sh]: create lmdb datasets
    - /home/share/shaofan/lfw\_caffe/train\_lmdb
    - /home/share/shaofan/lfw\_caffe/test\_lmdb
- [make\_mean.sh]: calculate the mean
    - /home/share/shaofan/lfw\_caffe/test\_lmdb
- [lfw\_cnn.py]: create and train a net 
    - solver.prototxt
    - /home/share/shaofan/lfw\_caffe/snapshot
