import os
os.chdir('../caffe')

import caffe
from caffe import layers as L
from caffe import params as P
import numpy as np

def lenet(lmdb, batch_size):
# our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
        transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)

    return n.to_proto()

if __name__ == '__main__':
    '''
        EXAMPLE=examples/mnist
        DATA=data/mnist
        BUILD=build/examples/mnist

        BACKEND="lmdb"

        echo "Creating ${BACKEND}..."

        rm -rf $EXAMPLE/mnist_train_${BACKEND}
        rm -rf $EXAMPLE/mnist_test_${BACKEND}

        $BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
          $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}
        $BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
          $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}
    '''

    with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_train_lmdb', 64)))

    with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
        f.write(str(lenet('examples/mnist/mnist_test_lmdb', 1000)))

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('examples/mnist/lenet_auto_solver.prototxt')

    print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
    print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

    solver.net.forward()  # train net
    print solver.test_nets[0].forward()  # test net (there can be more than one)

    solver.step(1)




    # Start to roll !

    niter = 200
    test_interval = 25
    # losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / test_interval)))
    output = np.zeros((niter, 8, 10))

    # the main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # accuracy
        solver.test_nets[0].forward()
        correct = (solver.test_nets[0].blobs['ip2'].data.argmax(1) == solver.test_nets[0].blobs['label'].data).sum()
        print '[%04d]: %.5f%%' % (it, correct/1e3*1e2)


        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
#       if it % test_interval == 0:
#           print 'Iteration', it, 'testing...'
#           correct = 0
#           for test_it in range(100):
#               solver.test_nets[0].forward()
#               correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
#           test_acc[it // test_interval] = correct / 1e4


