import os
import random

LMDB_ROOT = '/home/share/shaofan/lfw_caffe'
DATA_ROOT = '/home/share/lfw/'

def getlist(name):
    return map(lambda x: os.path.join(name, x), os.listdir(os.path.join(DATA_ROOT, name)) )

if __name__ == '__main__':
    people_names = os.listdir(DATA_ROOT)
    random.shuffle(people_names)

    separator = int(len(people_names) * 0.7)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for ind, ele in enumerate(people_names):
        if ind < separator:
            l = getlist(ele)
            train_x.extend(l)
            train_y.extend([ind]*len(l))
        else:
            l = getlist(ele)
            test_x.extend(l)
            test_y.extend([ind]*len(l))

#   l = zip(train_x, train_y)
#   random.shuffle(l)
#   train_x, train_y = zip(*l)
    
    with open('./train.txt', 'w') as f:
        for x, y in zip(train_x, train_y):
            f.write('%s %d\n' % (x, y))
    with open('./test.txt', 'w') as f:
        for x, y in zip(test_x, test_y):
            f.write('%s %d\n' % (x, y))
