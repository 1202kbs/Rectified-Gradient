import math

import tensorflow as tf
import numpy as np


class Trainer():

    def __init__(self, sess, model, data_train, batch_size=100):

        self.sess = sess
        self.model = model
        self.data_train = data_train
        self.batch_size = batch_size
    
    def train(self, n_epochs, p_epochs=10):
        
        for epoch in range(n_epochs):
            
            train_loss, train_acc = self.train_epoch(epoch)
            
            if (epoch + 1) % p_epochs == 0:
                
                print('Epoch : {:<3d} | Loss : {:.5f} | Train Accuracy : {:.5f}'.format(epoch + 1, train_loss, train_acc))
        
        self.model.save(self.sess)
    
    def train_epoch(self, epoch):

        avg_loss = 0
        avg_acc = 0
        n_itrs = math.ceil(len(self.data_train[0]) / self.batch_size)

        for itr in range(n_itrs):

            loss, acc = self.train_step(itr)
            avg_loss += loss / n_itrs
            avg_acc += acc / n_itrs
        
        return avg_loss, avg_acc

    def train_step(self, itr):

        batch_xs, batch_ys = self.data_train[0][itr * self.batch_size:(itr + 1) * self.batch_size], self.data_train[1][itr * self.batch_size:(itr + 1) * self.batch_size]

        feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys}
        _, loss, acc = self.sess.run([self.model.train, self.model.loss, self.model.accuracy], feed_dict=feed_dict)

        return loss, acc