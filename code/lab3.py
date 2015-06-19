import core.ga5
import core.loss_functions
from core.cnn import CNN, load_data
import matplotlib.pyplot as plt

import theano
import numpy as np
import cPickle

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.on_unused_input='ignore'

    all_datasets = cPickle.load(open('../../../Data/lfw/data.dat', 'rb'))
    size = all_datasets[0]
    print size 
    print all_datasets[1][0].shape
    print all_datasets[2][0].shape
    def share(a): 
        return (theano.shared(
                    np.asarray(
                        a[0].reshape(
                            a[0].shape[0], 2, size[0], size[1]),
                        dtype=theano.config.floatX
                    )
                ),
                theano.shared(
                    np.asarray(
                        a[1].flatten(),
                        dtype='int32'
                    )
                )
               )
                 
    datasets = (share(all_datasets[1]), share(all_datasets[1]))
    datasets_for_abstract = share(all_datasets[2])
 
    def examinate(cnn):
        x = datasets[0][0].eval()
        y = datasets[0][1].eval()
        k = cnn.nkerns[-1][0]
        shp = cnn.i_shp
        for i in [-1, 1]:
            cen = x[y==i]
            cen = cen.sum(0)/cen.shape[0]
            feature = cnn.get_feature(cen.reshape(1, 2, 32, 32))
        
            plt.subplot(2, 1, (i+1)/2)
            plt.imshow(feature.reshape(k, shp[0], shp[1])[0], interpolation='None', cmap='binary')
        plt.show()

    def dump2file(cnn, filename):
        global datasets_for_abstract
        x = datasets_for_abstract[0]
        y = datasets_for_abstract[1].eval()

        features = cnn.get_feature(x)

        print features.shape
        print y.shape
        cPickle.dump((features, y), open(filename, 'wb'))
        

    cnn = CNN(dim_in = 2, size_in = (32, 32), nkerns = [(8, (2, 2), (1, 1)), (6, (3, 3), (2, 2))])
    #fit(self, lossf, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):

#    examinate(cnn)
    dump2file(cnn, './features/1_random.feature')

    examinate(cnn)
    cnn.fit(core.loss_functions.TEST_LOSS_F, datasets, batch_size = 50, n_epochs = 100, learning_rate = 0.00005)
    examinate(cnn)

    dump2file(cnn, './features/1_trained.feature')




