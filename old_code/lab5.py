# lab5 simple cnn for lwf
from cPickle import dump, load
from numpy import *
from os import walk 
import cv2



N = 5749 
A = 50

def preload(path):
    x = []
    y = []
    cury = 0

    for dirpath, dirnames, filenames in walk(path):
        if filenames == []: continue
        for f in filenames:
            y.append(cury)
            x.append( cv2.resize( cv2.imread(dirpath+'/'+f, 0), (A, A)) )# gray 50*50
        cury += 1
        print '%.3f%%' % (float(cury)/N * 100)#5749. * 100)
        if cury >= N: break
    x = array(x, dtype=theano.config.floatX)
    y = array(y)

    dump((x, y), open('all_data.dat', 'wb'))
    raw_input('preload done')

def load_from_file():
    x, y = load(open('all_data.dat', 'rb'))
    return x, y

def direct_load(path, N=N):
    x = []
    y = []
    cury = 0

    for dirpath, dirnames, filenames in walk(path):
        if filenames == []: continue
        for f in filenames:
            y.append(cury)
            x.append( cv2.resize( cv2.imread(dirpath+'/'+f, 0), (A, A)) )# gray 50*50
        cury += 1
        print '%.3f%%' % (float(cury)/N * 100)#5749. * 100)
        if cury >= N: break
    x = array(x, dtype=theano.config.floatX)
    y = array(y)
    return x, y


import theano
import theano.tensor as T

def get_dataset(all_x, all_y, alpha=0.7, num = None):
    total_size = len(all_y)

# no shuffle
#    random_index = random.permutation(total_size)
#    all_x = all_x[random_index]
#    all_y = all_y[random_index]

    sep = int(total_size * alpha)
    train_set = (all_x[:sep], all_y[:sep])
    test_set = (all_x[sep:], all_y[sep:])


    def shared_dataset(data_xy, borrow=True, num = None):
        data_x, data_y = data_xy
        if num:
            data_x = data_x[:num]
            data_y = data_y[:num]

        shared_x = theano.shared(asarray(data_x,
                                 dtype=theano.config.floatX),
                                 borrow=True)
        shared_y = theano.shared(asarray(data_y,
                                 dtype=theano.config.floatX),
                                 borrow=True)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set, num = num)
#    valid_set_x, valid_set_y = shared_dataset(valid_set, num = num)
    train_set_x, train_set_y = shared_dataset(train_set, num = num)

    rval = [(train_set_x, train_set_y), #(valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


from core.mnist_mlp import cnn, raw_dump, raw_load

def continue_train(dataset):
    k = raw_load('model.dat')
    print k
    k.fit(dataset, n_epochs=100, learning_rate=0.000001)
    raw_dump(k, 'model.dat') 
    return k

def train_new(dataset):
    k = cnn(dim_in = 1, size_in = (A, A), size_out = N, 
            nkerns = [(20, (5, 5), (2, 2)), (15, (5, 5), (2, 2))], 
            nhiddens=[100])

    print k
    k.fit(dataset, n_epochs=100, learning_rate=0.0001)
    raw_dump(k, 'model.dat') 
    return k


def test_model(x, y, k=None):
    if k == None:
        k = raw_load('model.dat')
    print k

    size = y.shape[0]

    test_N = 5000
    diff_correct = 0
    d1, d2= [], []

    # diff test
    for _ in range(test_N): 
        d1_ind = random.randint(0, size)
        d2_ind = random.randint(0, size)
        while y[d1_ind] == y[d2_ind]:
            d2_ind = random.randint(0, size)

        d1.append(d1_ind)
        d2.append(d2_ind)

    y1 = k.pred(x[d1])
    y2 = k.pred(x[d2])
    diff_correct = (y1 != y2).sum()
    print 'diff accuracy: %d / %d = %.3f' % (diff_correct, test_N, float(diff_correct)/test_N)

    # same test
    same_correct = 0
    d1, d2, ans = [], [], []
    l = arange(size)
    for _ in range(test_N): 
        yk = random.randint(0, N)
        ans.append(yk)
        mask = l[y==yk]
        d1_ind = random.choice(mask)
        d2_ind = random.choice(mask)

        d1.append(d1_ind)
        d2.append(d2_ind)

    ans = array(ans)
    y1 = k.pred(x[d1])
    y2 = k.pred(x[d2])
    same_correct = (y1 == y2).sum()
    print 'same accuracy: %d / %d = %.3f ' % (same_correct, test_N, float(same_correct)/test_N)
    
    correct_correct = (y1 == ans).sum()
    print 'correct accuracy: %d / %d = %.3f ' % (correct_correct, test_N, float(correct_correct)/test_N)


def test_model_unseen(x, y, test_N=500, k=None):
    if k == None:
        k = raw_load('model.dat')

    size = y.shape[0]

    pred = k.pred(x)
    correct = 0
    sample_same = 0
    sample_diff = 0

    for _ in range(test_N):
        d1_ind = random.randint(0, size)
        d2_ind = random.randint(0, size)
        while d1_ind == d2_ind: 
            d2_ind = random.randint(0, size)

        if (pred[d1_ind]==pred[d2_ind]) is (y[d1_ind]==y[d2_ind]):
            correct += 1
        if y[d1_ind]==y[d2_ind]:
            sample_same += 1
        else:
            sample_diff += 1
    print 'Accuracy: %d / %d = %.3f' % (correct, test_N, float(correct)/test_N)
    print 'Sample = %d (same) + %d (diff)' % (sample_same, sample_diff)

    print k




import datetime
if __name__ == '__main__':
    #preload('/home/share/lfw/')
    #x, y = load_from_file()

    x, y = direct_load('/home/share/lfw/')

#    x = x.swapaxes(1, 3).swapaxes(2, 3)
    x = x.reshape(x.shape[0], 1, A, A)
    print x.shape, y.shape
    dataset = get_dataset(x, y)

#continue_train(dataset)

    #starttime = datetime.datetime.now()    
#    k = train_new(dataset)
    #endtime = datetime.datetime.now()    
    #print 'training taks %d seconds' % (endtime-starttime).seconds


    #starttime = datetime.datetime.now()    
    test_model(x, y)#, k)
    test_model_unseen(x, y)#, k)

    #endtime = datetime.datetime.now()    
    #print 'testing taks %d seconds' % (endtime-starttime).seconds




