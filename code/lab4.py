# lab 1
# two conv layers
# each with a fixed pool down layer 
# [(1~10, (1~10, 1~10), (2, 2)),
#  (1~10, (1~10, 1~10), (2, 2))]

import time
import multiprocessing

import core.ga5 as ga5
import core.loss_functions as loss_functions
import core.cnn as cnn

import test_features
import numpy as np
import gzip, cPickle
import theano
from svmutil import svm_train, svm_predict

log_file = open('./lab4/info.txt', 'a')
datasets = cnn.load_data('../../../Data/mnist/mnist.pkl.gz', 2000)
x, y = cnn.load_data('../../../Data/mnist/mnist.pkl.gz', 8000)[0]
y = list(y.eval())
x = x.eval()
l = int(len(y)*0.7)

def train_and_get(c, lr):
    c.reset_weight()
    c.compile_lr(lr)
    print '..building the cnn model: %r with lr: %r' % (c.nkerns, lr)
    c.fit_lr(n_epochs = 200, slient = True)

    features = test_features.tran2libsvm(c.get_feature(x))
    m = svm_train(y[:l], features[:l], '-q')
    p_label, p_acc, p_val = svm_predict(y[l:], features[l:], m)
    ans = p_acc[0]
    return ans

def cnn_fitness(base):
    nkerns = [(base[0], (base[1], base[1]), (2, 2)), (base[2], (base[3], base[3]), (2, 2))]
    c = cnn.CNN(dim_in = 1, size_in = (28, 28), nkerns = nkerns)
    if c.i_shp[0]<=0: return -np.inf
    c.set_lossf(loss_functions.TEST_LOSS_F)

    c.set_datasets(datasets, batch_size = 200) 

    ans = [train_and_get(c, i) for i in [0.1, 0.001, 0.000001]]
    print 'base: ', base 
    print 'ans list: ', ans
    log_file.write('base: %r, list: %r\n' % (base, ans))
    log_file.flush()

    # ans = train_and_get(0.0001) 
    return max(ans) 

def cnn_gen_init():
    rng = np.random.RandomState(int(time.time())+multiprocessing.current_process().pid)
    return rng.randint(1, 13, size=(4,))

def cnn_mutation(base):
    x = base+np.random.randint(-1, 1, size=(4,))
    x[x<=0] = 1
    return x 

def cnn_plot_f(base, clr='none', clean=False):
    i = [(base[0], (base[1], base[1]), (2, 2)), (base[2], (base[3], base[3]), (2, 2))]
    print i 
    log_file.write('ga plot: %r\n' % i)
    log_file.flush()

def cnn_crossover(father, mother):
    child0 = np.zeros(4,int) 
    child1 = np.zeros(4,int) 
    for i in range(4):
        if np.random.rand() > 0.5:
            child0[i] = father[i]
            child1[i] = mother[i]
        else:
            child1[i] = father[i]
            child0[i] = mother[i]
    return child0, child1

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'

    try:
        MyGA = ga5.GA(cnn_fitness, {'iter_maximum':300}, 
                        12, cnn_gen_init,
                        0.70, cnn_crossover, 0.15, cnn_mutation, 
                        True, cnn_plot_f, 1, lambda a,b: a>b,
              cores = 4)
        MyGA.fit()
    finally:
        log_file.close()


#   def __init__(self, fitness_f, terminator, 
#       generation_size, generation_init, 
#       p_crossover, crossover_f, mutation_rate, mutation_f, 
#       plot = False, plot_f = None, plot_interval = 100,
#       cmp_f = None,
#       cores = 1 ):

