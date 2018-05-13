#coding=utf-8

print("0")
import tensorflow as tf
print("1")

import tensorflow.examples.tutorials.mnist.input_data
# 加载数据
# mnist = input_data.read_data_sets(r'./mnist', one_hot=True)
"""
# 创建模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
"""
print("2")

import struct
from glob import glob
import os
import numpy as np 

print("2")
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]
    print(images_path,images_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
        x=np.zeros((labels.shape[0],10))
        for i in range(labels.shape[0]):
            x[i][labels[i]]=1
        
        # for i in range(x.shape[0]):
        #     print(np.argmax(x[i]),np.argmax(x[i])==labels[i])
        labels=np.array(x)

        

        
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
        print(images.shape)
    return images, labels

images, labels = load_mnist('./mnist')
test_images, test_labels = load_mnist('./mnist', 't10k')

x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
W2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.matmul(layer1, W2) + b2

# 正确的样本标签
y_ = tf.placeholder(tf.float32, [None, 10])

# 损失函数选择softmax后的交叉熵，结果作为y的输出
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print("now")
# 训练过程
batch_size=10
for i in range(5000):
    batch_xs=images[i * batch_size:(i + 1) * batch_size]
    batch_ys = labels[i * batch_size:(i + 1) * batch_size]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i %100 == 0:
        # 使用测试集评估准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print (sess.run(accuracy, feed_dict = {x: test_images,
                                                  y_: test_labels}))