import os
import random

LMDB_ROOT = '/home/share/shaofan/lfw_caffe'
DATA_ROOT = '/home/share/lfw/'

def getlist(name):
    return map(lambda x: os.path.join(name, x), os.listdir(os.path.join(DATA_ROOT, name)) )

def getall():
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
    return test_x, test_y, train_x, train_y

def getFromList(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        a = int(f.readline().strip())
        for label in range(a):
            line = f.readline()
            name, num = line.strip().split('\t')
            for ind in range(int(num)):
                x.append(os.path.join(name, '%s_%04d.jpg'%(name, ind+1)) )
                y.append(label)
    return x, y

if __name__ == '__main__':
#   all y is generated as the multiclass classify target

# get all data
#    test_x, test_y, train_x, train_y = getall()

    x, y = getFromList('./lfw_rules/peopleDevTrain.txt')
    
    zipped = zip(x, y)
    random.shuffle(zipped)
    x, y = zip(*zipped)
    sep = int(len(y) * 0.7)
    train_x, train_y, test_x, test_y = x[:sep], y[:sep], x[sep:], y[sep:]

    with open('./train.txt', 'w') as f:
        for x, y in zip(train_x, train_y):
            f.write('%s %d\n' % (x, y))
    with open('./test.txt', 'w') as f:
        for x, y in zip(test_x, test_y):
            f.write('%s %d\n' % (x, y))

    print 'Done.'
