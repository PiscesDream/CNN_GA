import cPickle
import numpy as np
from svmutil import svm_train, svm_predict

def tran2libsvm(a):
    l = a.shape[1]
    r = range(1, l+1)
    return map(lambda x: dict(zip(r, x)), a)  

def import_data(filename):
    print 'loading %s ...' % filename
    features, y = cPickle.load(open(filename, 'r'))
    print '\tfeatures: ', features.shape
    print '\ty: ', y.shape

    return tran2libsvm(features), list(y)

def testfeature(filename):
    x, y = import_data(filename)
    l = int(len(y)*0.7)
    m = svm_train(y[:l], x[:l], '')
    p_label, p_acc, p_val = svm_predict(y[l:], x[l:], m)
    print p_acc

if __name__ == '__main__':
    num = 1
    testfeature('./features/%d_random.feature' % num)
    raw_input('pause')
    testfeature('./features/%d_trained.feature' % num)

