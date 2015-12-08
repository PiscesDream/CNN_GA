import os
import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np
import gc

LMDB_ROOT = '/home/share/shaofan/lfw_caffe/lmdb'
MULTICLASS_LMDB = os.path.join(LMDB_ROOT, 'multiclass')
TRAIN_LMDB = os.path.join(LMDB_ROOT, 'bin_train')
TEST_LMDB= os.path.join(LMDB_ROOT, 'bin_test')
BATCH_SIZE = 100

def getMulticlassNet(lmdb, batch_size):
    # LeNet for lfw
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
        transform_param=dict(scale=1./255), ntop=2)
    n.conv = L.Convolution(n.data, kernel_size=10, num_output=10, weight_filler=dict(type='xavier'))
    n.pool = L.Pooling(n.conv, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool, num_output=100, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=4000, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

def getBinclassNet(lmdb, batch_size=BATCH_SIZE):
    # LeNet for lfw
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
        transform_param=dict(scale=1./255), ntop=2)
    n.pic1, n.pic2 = L.Slice(n.data, slice_param={'slice_dim':1, 'slice_point':3}, ntop=2)

    n.conv1 = L.Convolution(n.pic1, kernel_size=10, num_output=10, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.conv2 = L.Convolution(n.pic2, kernel_size=10, num_output=10, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    n.diff = L.Eltwise(n.pool1, n.pool2, eltwise_param={'operation': 1, 'coeff': [1, -1]})
    n.fc = L.InnerProduct(n.diff, num_output=2, weight_filler={'type': 'xavier'})
    n.loss = L.SoftmaxWithLoss(n.fc, n.label)

    return n.to_proto()

def copyParams(net1, net2, layer_pairs):
    for l1, l2 in layer_pairs:
        net1.params[l1][0].data.flat = net2.params[l2][0].data.flat
        net1.params[l1][1].data.flat = net2.params[l2][1].data.flat

def createSolver(filename, **kwargs):
    with open(filename, 'w') as f:
        for key, val in kwargs.iteritems():
            f.write('{}: {}\n'.format(key, val))
    return filename

def runSolver(solver, predict_layer, test_set=False, N=10):
    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

    test_interval = 100
    # losses will also be stored in the log

    # the main solver loop
    for it in range(N):
        # accuracy
        correct = 0
        for i in xrange(10):
            solver.net.forward()
            correct += (solver.net.blobs[predict_layer].data.argmax(1) == solver.net.blobs['label'].data).sum()
        print 'Train: [%04d]: %04d/1000=%.5f%%' % (it, correct, correct/1e3*1e2)

        if test_set:
            correct = 0
            for i in xrange(10):
                solver.test_nets[0].forward()
                correct += (solver.test_nets[0].blobs['fc'].data.argmax(1) == solver.net.blobs['label'].data).sum()
            print 'Test: [%04d]: %04d/1000=%.5f%%' % (it, correct, correct/1e3*1e2)

        solver.step(test_interval)  # SGD by Caffe



if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()

    with open('./lfw_multiclass.prototxt', 'w') as f:
        f.write(str(getMulticlassNet(MULTICLASS_LMDB, 100)))
    multiclass_solver = caffe.SGDSolver(
        createSolver('multiclassSolver.prototxt', 
            train_net='"./lfw_multiclass.prototxt"',
            base_lr=0.010,
            momentum=0.9,
            weight_decay=0.001,
            lr_policy='"inv"',
            gamma=0.001,
            power=0.75,
            display=10,
            solver_mode="GPU"))
    runSolver(multiclass_solver, 'ip2', N=20)


    with open('./lfw_train.prototxt', 'w') as f:
        f.write(str(getBinclassNet(TRAIN_LMDB)))
    with open('./lfw_test.prototxt', 'w') as f:
        f.write(str(getBinclassNet(TEST_LMDB)))
    binclass_solver = caffe.SGDSolver(
        createSolver('binclassSolver.prototxt', 
            train_net='"./lfw_train.prototxt"',
            test_net='"./lfw_test.prototxt"',
            test_iter=1,
            test_interval=100,
            base_lr=0.001,
            momentum=0.9,
            weight_decay=0.00005,
            lr_policy='"inv"',
            gamma=0.001,
            power=0.75,
            display=10,
            snapshot=2000, 
            snapshot_prefix='"/home/share/shaofan/lfw_caffe/snapshot/"',
            solver_mode="GPU"))

    print('copying params ...')
    copyParams(binclass_solver.net, multiclass_solver.net, 
        [('conv1', 'conv'),('conv2', 'conv')])
    assert((binclass_solver.net.params['conv1'][0].data.flat==binclass_solver.net.params['conv2'][0].data.flat).all())
    assert((binclass_solver.net.params['conv1'][1].data.flat==binclass_solver.net.params['conv2'][1].data.flat).all())

    multiclass_solver = None
    gc.collect()


    binclass_solver.net.forward()
    print binclass_solver.net.blobs['data'].data.shape
    print binclass_solver.net.blobs['pic1'].data.shape
    print binclass_solver.net.blobs['diff'].data.shape
    print binclass_solver.net.blobs['fc'].data.shape

    runSolver(binclass_solver, 'fc', test_set=True, N=20)
