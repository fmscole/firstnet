import struct
from glob import glob
import os
import numpy as np
class my_data_set:
    def __init__(self, images, labels):
        self.images=images
        self.labels=labels
        self.size=images.shape[0]
        self.i=0

    def next_batch(self,batch_size):
        i=self.i
        if i + batch_size<=self.size:
            batch_xs1 = self.images[i :i +  batch_size]
            batch_ys1 = self.labels[i :i + batch_size]
            self.i=self.i+batch_size
            if self.i==self.size:
                self.i=0
            return batch_xs1,batch_ys1
        if i <self.size:
            batch_xs1 = self.images[i:]
            batch_ys1 = self.labels[i :]
            self.i =batch_size-self.size+self.i
            batch_xs1 = np.concatenate([batch_xs1,self.images[0:self.i ]],axis=0)
            batch_ys1 =np.concatenate([batch_ys1,self.labels[0:self.i ]],axis=0)
            print (self.i,batch_xs1.shape[0])
            return batch_xs1,batch_ys1

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]
    print(images_path, images_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))

        labels = np.fromfile(lbpath, dtype=np.uint8)
        x = np.zeros((labels.shape[0], 10))
        for i in range(labels.shape[0]):
            x[i][labels[i]] = 1
        labels = np.array(x)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        images=np.array(images)/255
        # print(images.shape)
    return images, labels



if __name__=="__main__":

    x=np.zeros((14,))
    a=my_data_set(x,x)

    for i in range(1000):
        a.next_batch(13)