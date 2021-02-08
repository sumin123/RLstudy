import tensorflow as tf
import numpy as np

class Qnet():
    def __init__(self, session, state_dim, action_dim, learning_rate, name):
        self.gamma = 0.98
        self.state_dim = state_dim
        self.n_action = action_dim
        self.lr = learning_rate
        self.sess = session

        self.build_network(name)

    def build_network(self, name):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, self.state_dim], name='X')
            net = self.X
            net = tf.layers.dense(net, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            net = tf.layers.dense(net, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            net = tf.layers.dense(net, self.n_action, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            self.Q = net

            self.Y = tf.placeholder(tf.float32, [None, self.n_action], name='Y')
            self.loss = tf.losses.mean_squared_error(self.Y, self.Q)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.train = optimizer.minimize(self.loss, var_list=var)

    def sample_action(self, state):
        Q = self.sess.run(self.Q, feed_dict={self.X: state})
        return Q

    def update(self, X, Y):
        return self.sess.run([self.loss, self.train, self.Q, self.Y], feed_dict={self.X: X, self.Y: Y})