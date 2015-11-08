# lab 6
# [(1~10, (1~10, 1~10), (2, 2)),
#  (1~10, (1~10, 1~10), (2, 2)),
# fc: 500

import time
import multiprocessing

import core.ga5 as ga5
import core.loss_functions as loss_functions
from numpy import random, arange, array

import numpy as np
import gzip, cPickle
import theano


from lab5 import direct_load, get_dataset
from core.mnist_mlp import cnn

log_file = open('./lab6/info.txt', 'a')

N = 5000 
A = 50
x, y = direct_load('/home/share/lfw/', N)
x = x.reshape(x.shape[0], 1, A, A)
print x.shape, y.shape
datasets = get_dataset(x, y)

#prepare test
size = y.shape[0]

test_N = 500
diff_correct = 0
same_correct = 0
d1, d2= [], []
# diff test
for _ in range(test_N): 
    d1_ind = random.randint(0, size)
    d2_ind = random.randint(0, size)
    while y[d1_ind] == y[d2_ind]:
        d2_ind = random.randint(0, size)

    d1.append(d1_ind)
    d2.append(d2_ind)
test_diff1 = x[d1]
test_diff2 = x[d2]

# same test
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
test_same1 = x[d1]
test_same2 = x[d2]
test_same_ans = array(ans)

def test_model(k):
    y1 = k.pred(test_diff1)
    y2 = k.pred(test_diff2)
    diff_correct = (y1 != y2).sum()
    #print '\tdiff accuracy: %d / %d' % (diff_correct, test_N)

    y1 = k.pred(x[d1])
    y2 = k.pred(x[d2])
    same_correct = (y1 == y2).sum()
    #print '\tsame accuracy: %d / %d' % (same_correct, test_N)
    return float(same_correct+diff_correct)/(test_N*2.0)

def train_and_get(k, lr):
    print '..building the cnn model: %r with lr: %r' % (k.nkerns, lr)
    k.fit(datasets, n_epochs=100, learning_rate=lr)
    return test_model(k)

def cnn_fitness(base):
    nkerns = [(base[0], (base[1], base[1]), (2, 2)), (base[2], (base[3], base[3]), (2, 2))]
    k = cnn(dim_in = 1, size_in = (A, A), size_out=N, nkerns = nkerns, nhiddens=[500])
    ans = [train_and_get(k, i) for i in [0.1, 0.001, 0.000001]]
    print '\tnkerns: ', nkerns 
    print '\tans list: ', ans
    log_file.write('base: %r, list: %r\n' % (base, ans))
    log_file.flush()

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
                        10, cnn_gen_init,
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

