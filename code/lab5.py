# lab5 simple cnn for lwf
from numpy import *
from os import walk 
import cv2

N = 20

def load_all(path):
    x = []
    y = []
    cury = 0

    for dirpath, dirnames, filenames in walk(path):
        if filenames == []: continue
        for f in filenames:
            y.append(cury)
            x.append( cv2.imread(dirpath+'/'+f) )
        cury += 1
        print '%.3f%%' % (float(cury)/N * 100)#5749. * 100)
        if cury >= N: break
    x = array(x)
    y = array(y)
    return x, y

import theano
import theano.tensor as T

def get_dataset(all_x, all_y, alpha=0.7, num = None):
    total_size = len(all_y)

    random_index = random.permutation(total_size)
    all_x = all_x[random_index]
    all_y = all_y[random_index]

    sep = int(total_size * alpha)
    test_set = (all_x[sep:], all_y[sep:])
    train_set = (all_x[:sep], all_y[:sep])


    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]

        shared_x = theano.shared(asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(asarray(data_y,
                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set, num = num)
#    valid_set_x, valid_set_y = shared_dataset(valid_set, num = num)
    train_set_x, train_set_y = shared_dataset(train_set, num = num)

    rval = [(train_set_x, train_set_y), #(valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


from core.mnist_mlp import cnn

if __name__ == '__main__':
    x, y = load_all('../../../Data/lfw/lfw/')
    x = x.swapaxes(1, 3).swapaxes(2, 3)
    print x.shape, y.shape

    k = cnn(dim_in = 3, size_in = (250, 250), size_out = N, 
#            nkerns = [(10, (5, 5), (2, 2)), (16, (5, 5), (2, 2))], 
            nhiddens=[40, 20])

    dataset = get_dataset(x, y)
    k.fit(dataset)
    

    




