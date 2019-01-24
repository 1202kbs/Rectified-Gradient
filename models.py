import os, math

import tensorflow as tf


class CIFAR_CNN:
    
    def __init__(self, logdir, name):
        
        if not os.path.exists(logdir):
            
            os.makedirs(logdir)
        
        self.logdir = logdir
        self.name = name
        
        self.build_model()
        self.init_saver()
    
    def save(self, sess):

        self.saver.save(sess, self.logdir + 'model')

    def load(self, sess):

        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)

        if latest_checkpoint:

            self.saver.restore(sess, latest_checkpoint)
    
    def evaluate(self, sess, dataset):

        batch_size = 100
        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        avg_acc = 0

        for itr in range(n_itrs):

            batch_xs, batch_ys = dataset[0][itr * batch_size:(itr + 1) * batch_size], dataset[1][itr * batch_size:(itr + 1) * batch_size]

            feed_dict = {self.X: batch_xs, self.Y: batch_ys}
            acc = sess.run(self.accuracy, feed_dict=feed_dict)
            avg_acc += acc / n_itrs

        return avg_acc
    
    def build_model(self):
        
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X')
        self.Y = tf.placeholder(tf.int64, [None], 'Y')
        Y_hot = tf.one_hot(self.Y, depth=10)
        
        with tf.variable_scope(self.name):
            
            conv1 = tf.layers.conv2d(inputs=self.X, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=[2,2], padding='SAME', name='pool2')
            
            conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv3')
            conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu, use_bias=True, name='conv4')
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=[2,2], padding='SAME', name='pool4')
            flat4 = tf.reshape(pool4, [-1, 8 * 8 * 64], name='flat4')
            
            dense5 = tf.layers.dense(inputs=flat4, units=256, activation=tf.nn.relu, use_bias=True, name='dense5')
            self.logits = tf.layers.dense(inputs=dense5, units=10, use_bias=True, name='dense6')
        
        tf.add_to_collection('tensors', self.X)
        tf.add_to_collection('tensors', self.logits)
        
        predictions = tf.argmax(self.logits, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(Y_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='Accuracy')
        
        self.yi = tf.argmax(self.logits, 1, name='Prediction')
        self.yx = tf.nn.softmax(self.logits, name='Scores')
        self.yv = tf.reduce_max(self.logits, 1, name='MaxScore')
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=Y_hot))
        self.train = tf.train.AdamOptimizer().minimize(self.loss, var_list=self.vars)
    
    def init_saver(self):
        
        self.saver = tf.train.Saver()
    
    @property
    def vars(self):
        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)