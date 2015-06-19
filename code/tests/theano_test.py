import theano.tensor as T
import numpy.random as random
import theano

import numpy as np 
from time import sleep

class poly(object):
    def __init__(self, coef):
        coef = [0] if coef is None else coef
        l = len(coef)
        self.coef = theano.shared(np.asarray(coef, dtype=theano.config.floatX))
        self.velocity = theano.shared(np.random.rand(l,))

        self.x = T.scalar('x')
        self.output = 0
        for i in range(0, l):
            self.output = self.output * self.x + self.coef[i]

    def __repr__(self):
        coef = self.coef.eval()
        return 'poly: %r' % coef

    def fit(self, lossf, n_epochs):
        grads = T.grad(lossf, self.coef)
#        grads = grads/T.max(T.abs_(grads))
#        self.velocity = 0.1 * self.velocity - grads * 0.9
#        self.velocity = self.velocity/T.sum(self.velocity)
        updates = [(self.coef, self.coef - grads)]

        train_model = theano.function([self.x], lossf, 
            updates = updates)

        print '...training'
        maxiter = n_epochs
        iteration = 0
        while iteration < maxiter:
            iteration += 1
            print 'iteration %d' % iteration
            for x in np.linspace(-10, 10, 50):
                print '\tL of f(%.4f) = %.4f\r' % (x, train_model(x)),
	    print ''
        print self


if __name__ == '__main__':
    ma = poly([1,2,3])

    delta = (ma.output - 10) ** 2

    ma.fit(delta, 1000)


