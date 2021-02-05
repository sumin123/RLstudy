import tensorflow as tf
import numpy as np

class Qnet():
    def __init__(self, session, state_dim, action_dim, learning_rate, name):
        self.gamma = 0.98
        self.state_dim = state_dim
        self.n_action = action_dim
        self.X = tf.placeholder(tf.float32, [None, self.state_dim], name='X')
        self.Y = tf.placeholder(tf.float32, [None, self.n_action], name='Y')
        self.lr = learning_rate
        self.sess = session

        self.build_network(name)

    def build_network(self, name):
        with tf.variable_scope(name):
            net = self.X
            net = tf.layers.dense(net, 128, activation=tf.nn.relu)
            net = tf.layers.dense(net, 128, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.n_action)
            self.Q = tf.layers.dense(net, self.n_action, activation=None)
            self.loss = tf.losses.mean_squared_error(self.Y, self.Q)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train = optimizer.minimize(self.loss)

    def sample_action(self, state):
        Q = self.sess.run(self.Q, feed_dict={self.X: state})
        return Q

    def update(self, s, a, r, s_prime, done_mask):
        '''
        q = self.sample_action(s)
        a = np.array(a)
        q_a = np.take_along_axis(q, a, axis=1)
        '''
        target = r + self.gamma * np.max(self.sample_action(s_prime), axis=1) * done_mask
        q_target = self.sample_action(s)
        q_target[np.arange(len(s)), a] = target

        return self.sess.run([self.loss, self.train], feed_dict={self.X: s, self.Y: q_target})