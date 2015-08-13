import os
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

LMDB_ROOT = '/home/share/shaofan/lfw_caffe'
TRAIN_LMDB = os.path.join(LMDB_ROOT, 'train_lmdb')
TEST_LMDB= os.path.join(LMDB_ROOT, 'test_lmdb')

def lenet(lmdb, batch_size):
    # LeNet for lfw
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
        transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=15, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=3, num_output=20, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=100, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=4000, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    return n.to_proto()

if __name__ == '__main__':
    with open('lfw_train.prototxt', 'w') as f:
        f.write(str(lenet(TRAIN_LMDB, 100)))

    with open('lfw_test.prototxt', 'w') as f:
        f.write(str(lenet(TEST_LMDB, 1000)))

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('./solver.prototxt')

    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

    solver.net.forward()  # train net



    # Start to roll !

    niter = 1000
    test_interval = 100
    # losses will also be stored in the log

    # the main solver loop
    for it in range(niter):
        solver.step(test_interval)  # SGD by Caffe

        # accuracy
        correct = 0
        for i in xrange(10):
            solver.net.forward()
#            print solver.net.blobs['ip2'].data.argmax(1)
            correct += (solver.net.blobs['ip2'].data.argmax(1) == solver.net.blobs['label'].data).sum()
        print '[%04d]: %04d/1000=%.5f%%' % (it, correct, correct/1e3*1e2)


