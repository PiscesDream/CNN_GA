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
    def share(a, num): 
        return (theano.shared(
                    np.asarray(
                        a[0][num].reshape(a[0].shape[1], 1, size[0], size[1]),
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
                 
    datasets0 = (share(all_datasets[1], 0), share(all_datasets[1], 0))
    datasets1 = (share(all_datasets[1], 1), share(all_datasets[1], 1))

    datasets_for_abstract0 = share(all_datasets[2], 0)
    datasets_for_abstract1 = share(all_datasets[2], 1)
 
    def examinate(cnn1, cnn2):
        x0 = datasets0[0][0].eval()
        x1 = datasets1[0][0].eval()

#       plt.subplot(2, 1, 1)
#       plt.imshow(x0[0][0], interpolation='None', cmap='gray')
#       plt.subplot(2, 1, 2)
#       plt.imshow(x1[0][0], interpolation='None', cmap='gray')
#       plt.show()

        y = datasets0[0][1].eval()
        k = cnn1.nkerns[-1][0]
        shp = cnn1.i_shp

        feature0 = cnn1.get_feature(x0).sum(0)
        plt.subplot(2, 1, 1)
        plt.imshow(feature0.reshape(k, shp[0], shp[1])[0], interpolation='None', cmap='gray')
        feature1 = cnn2.get_feature(x1).sum(0)
        plt.subplot(2, 1, 2)
        plt.imshow(feature1.reshape(k, shp[0], shp[1])[0], interpolation='None', cmap='gray')
        plt.show()

    def dump2file(cnn0, cnn1, filename):
        global datasets_for_abstract
        x0 = datasets_for_abstract0[0]
        x1 = datasets_for_abstract1[0]
        y = datasets_for_abstract0[1].eval()

        features0 = cnn1.get_feature(x0)
        features1 = cnn2.get_feature(x1)
        #features = np.concatenate([features0, features1], 1)
        features = features0 - features1  

        cPickle.dump((features, y), open(filename, 'wb'))
        

    cnn1 = CNN(dim_in = 1, size_in = (32, 32), nkerns = [(8, (2, 2), (1, 1)), (6, (3, 3), (2, 2))])
    cnn2 = CNN(dim_in = 1, size_in = (32, 32), nkerns = [(8, (2, 2), (1, 1)), (6, (3, 3), (2, 2))])
    #fit(self, lossf, datasets, batch_size = 500, n_epochs = 200, learning_rate = 0.01):

#    examinate(cnn)

    examinate(cnn1, cnn2)
    dump2file(cnn1, cnn2, './features/1_random.feature')

    for i in range(4):
        print 'The %03dth updating' % i
        loss = core.loss_functions.lossf3(cnn1, cnn2.get_feature(datasets1[0][0]))
        cnn1.set_lossf(loss)
        cnn1.fit(datasets0, batch_size = 50, n_epochs = 5, learning_rate = 0.00001, test_model_on = 0)

        loss = core.loss_functions.lossf3(cnn2, cnn1.get_feature(datasets1[0][0]))
        cnn2.set_lossf(loss)
        cnn2.fit(datasets1, batch_size = 50, n_epochs = 5, learning_rate = 0.00001, test_model_on = 0)

    examinate(cnn1, cnn2)
    dump2file(cnn1, cnn2, './features/1_trained.feature')




