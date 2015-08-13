import theano.tensor as T
import numpy as np
import theano

def lossf1(self, classes):
    '''
        the sum of the distance of each pair of all classes' centroid

            dist(i, j) = ||i - j||  
            loss = sigma_{i!=j} { dist(i, j) }
        where
            i, j belong to {1, 2, .., k} 
            k is the total classes in the problem
    '''
    cens = [] 
    for i in classes:
        child = self.output[T.eq(i, self.y).nonzero()]  #.reshape(-1, 10, 13, 13)  )
        cens.append( T.sum(child, 0)/child.shape[0] )
#            cens = kinds[i]/sums[i].shape[0] 

    diff = 0 
    for iind, i in enumerate(classes):
        for jind, j in enumerate(classes):
            if jind >= iind: continue
            diff += T.sum((cens[iind] - cens[jind]) ** 2)
    return -diff  


def lossf2(self, classes):
    '''
        the minimum of the distance of each pair of all classes' centroid

            dist(i, j) = ||i - j||  
            loss = min_{i!=j} { dist(i, j) }
        where
            i, j belong to {1, 2, .., k} 
            k is the total classes in the problem
    '''
    cens = [] 
    for i in classes:
        child = self.output[T.eq(i, self.y).nonzero()]  #.reshape(-1, 10, 13, 13)  )
        cens.append( T.sum(child, 0)/child.shape[0])
#            cens = kinds[i]/sums[i].shape[0] 

    diff = []
    for i in classes:
        for j in classes:
            if j >= i: continue
            diff.append( T.sum((cens[i] - cens[j]) ** 2) )
    return -T.min(T.stacklists(diff))


def lossf3(c1, compare):
    '''
        loss = dist * y
    '''
    batch_size = 50
    compare = theano.shared(compare)[c1.index*batch_size:(c1.index+1)*batch_size]
    return T.sum(  T.sum( ((c1.output - compare)**2), 1 ) - c1.y.flatten()  )



def default_lossf(self):        
    #build cost
#        diff = ((y - hid_layers[-1].output) ** 2).sum()
#        cost = diff + L2 * lmbd
    #log                   T.sum or T.mean is judged
    negative_likelihood = -T.sum(T.log(hid_layers[-1].output)[T.arange(y.shape[0]), y]) #y without boarden
    cost = negative_likelihood + L2 * lmbd

#        errors = T.mean(T.neq(T.argmax(y, 1), T.argmax(hid_layers[-1].output, 1)))
    y_pred = T.argmax(hid_layers[-1].output, axis = 1) 
    errors = T.mean(T.neq(y_pred, y))

    #build update
    params = []
    for ind, ele in enumerate(cnn_layers + hid_layers):
        params.extend(ele.params)
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - lr * grad_i))




classes = range(10)
#classes = [1, -1] 
TEST_LOSS_F = lambda x: lossf1(x, classes)
#TEST_LOSS_F = lossf3 

