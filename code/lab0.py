import core.ga5
import core.loss_functions
from core.cnn import CNN, load_data
import matplotlib.pyplot as plt

import theano
import cPickle

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'
    size = 28

    datasets = load_data('../../../Data/mnist/mnist.pkl.gz', 1000)
    datasets_for_abstract = load_data('../../../Data/mnist/mnist.pkl.gz', 5000)
 
    def examinate(cnn):
        x = datasets[0][0].eval()
        y = datasets[0][1].eval()
        k = cnn.nkerns[-1][0]
        shp = cnn.i_shp
        for i in range(10):
            cen = x[y==i]
            cen = cen.sum(0)/cen.shape[0]
            feature = cnn.get_feature(cen.reshape(1, 1, 28, 28))
        
            plt.subplot(2, 5, i+1)
            plt.imshow(feature.reshape(k, shp[0], shp[1])[0], interpolation='None', cmap='binary')
        plt.show()

    def dump2file(cnn, filename):
        global datasets_for_abstract
        x = datasets_for_abstract[0][0]
        y = datasets_for_abstract[0][1].eval()

        features = cnn.get_feature(x)

        print features.shape
        print y.shape
        cPickle.dump((features, y), open(filename, 'wb'))
        

    cnn = CNN(dim_in = 1, size_in = (28, 28), nkerns = [(4, (2, 2), (1, 1)), (2, (3, 3), (2, 2))])
    #fit(self, lossf, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):

#    examinate(cnn)
#    dump2file(cnn, './features/1_random.feature')

    examinate(cnn)
    cnn.set_lossf(core.loss_functions.TEST_LOSS_F)
    cnn.fit(datasets, batch_size = 200, n_epochs = 100, learning_rate = 0.0001)
    examinate(cnn)

#    dump2file(cnn, './features/1_trained.feature')



