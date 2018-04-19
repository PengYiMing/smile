import os
from os.path import isdir

import cv2 as cv
import numpy as np
import tensorflow as tf

def init_dataset():
    X_train = np.ndarray(shape=(1400, 32, 64, 1), dtype=np.float32)
    Y_train = np.ndarray(shape=(1400, 1), dtype=np.float32)
    X_test = np.ndarray(shape=(160, 32, 64, 1), dtype=np.float32)
    Y_test = np.ndarray(shape=(160, 1), dtype=np.float32)
    return X_train, Y_train, X_test, Y_test

def load_img(dir_path, dir, ind, X, Y):
    imgs = os.listdir(dir_path)
    for img in imgs:
        if img == '.DS_Store':
            continue
        img_path = dir_path + '/' + img
        img_array = cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE) / 255
        X[ind] = img_array[:, :, np.newaxis]
        Y[ind] = 1 if dir == 'img_smile' else 0
        ind += 1
    return ind

def load_dataset(dataset, X, Y):
    path = '../data/' + dataset
    dirs = os.listdir(path)
    ind = 0
    for dir in dirs:
        dir_path = path + '/' + dir
        if isdir(dir_path):
            ind = load_img(dir_path, dir, ind, X, Y)

def init_placeholder(n_h0, n_w0):
    X_holder = tf.placeholder(dtype=tf.float32, shape=(None, n_h0, n_w0, 1), name="X_holder")
    Y_holder = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    return X_holder, Y_holder

def init_conv_param():
    w1 = tf.get_variable('w1', [5, 5, 1, 2], initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2', [3, 3, 2, 4], initializer=tf.contrib.layers.xavier_initializer())
    return {'w1':w1, 'w2':w2}

def model(alpha=0.009, epoch=300):
    tf.reset_default_graph()

    X_train, Y_train, X_test, Y_test = init_dataset()
    load_dataset('train', X_train, Y_train)
    load_dataset('test', X_test, Y_test)

    _, n_h0, n_w0, _ = X_train.shape
    X_holder, Y_holder = init_placeholder(n_h0, n_w0)

    # forward propagation
    W = init_conv_param()
    # conv
    Z1 = tf.nn.conv2d(X_holder, filter=W['w1'], strides=[1, 2, 2, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    # max-pool
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv
    Z2 = tf.nn.conv2d(P1, filter=W['w2'], strides=[1, 2, 2, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    # max-pool
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # fully-connected
    P2 = tf.layers.flatten(P2)
    A3 = tf.contrib.layers.fully_connected(P2, num_outputs=10)
    # fully-connected(with linear activation)
    Z4 = tf.contrib.layers.fully_connected(A3, num_outputs=1, activation_fn=None)
    Y_predict = tf.sigmoid(Z4, name='Y_predict')

    # cost & optimizer
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(Y_holder, Y_predict))
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

    # start training
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(epoch + 1):
            cur_cost, _ = sess.run([cost, optimizer], feed_dict={X_holder:X_train, Y_holder:Y_train})
            if e % 5 == 0:
                print('epoch = %d, cost = %f' % (e, cur_cost))

        # predict result
        correct_prediction = tf.equal(tf.equal(Y_holder, 1), tf.greater_equal(Y_predict, 0.6))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X_holder: X_train, Y_holder: Y_train})
        test_accuracy = accuracy.eval({X_holder: X_test, Y_holder: Y_test})
        print("train accuracy:", train_accuracy)
        print("test accuracy:", test_accuracy)

        # save model
        tf.train.Saver().save(sess, './smile-model', global_step=epoch)

if __name__ == '__main__':
    model()

