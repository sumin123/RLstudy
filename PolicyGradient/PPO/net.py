import tensorflow as tf
import numpy as np


class Net():
    def __init__(self):
        super(Net, self).__init__()
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.lr = 0.0005
        self.K = 2

        self.build_network('net')
        self.build_net_train()

    def build_network(self, name):
        ob_space = 4
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, ob_space], name='X')

            with tf.variable_scope('policy_net'):
                layer1 = tf.layers.dense(inputs=self.X, units=256, activation=tf.nn.relu)
                self.action_prob = tf.layers.dense(inputs=layer1, units=2, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer1 = tf.layers.dense(inputs=self.X, units=256, activation=tf.nn.relu)
                self.v_pred = tf.layers.dense(inputs=layer1, units=1, activation=None)

            self.action = tf.argmax(self.action_prob, axis=1)
            self.action_prob_select = tf.reduce_max(self.action_prob, axis=1)

            self.scope = tf.get_variable_scope().name

    def build_net_train(self):
        with tf.variable_scope('train_var'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='action')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
            self.action_prob_old = tf.placeholder(dtype=tf.float32, shape=[None], name='action_prob_old')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        with tf.variable_scope('loss_pi'):
            ratio = tf.exp(tf.log(self.action_prob_select) - tf.log(self.action_prob_old))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratio), tf.multiply(self.gaes, clipped_ratio))
            loss_clip = tf.reduce_sum(loss_clip)

        with tf.variable_scope('loss_v'):
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None], name='td_target')
            loss_v = tf.squared_difference(self.td_target, self.v_pred)
            loss_v = tf.reduce_sum(loss_v)

        with tf.variable_scope('loss'):
            self.loss = - loss_clip + 1 * loss_v

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
        model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.train_op = optimizer.minimize(self.loss, var_list=model_var)

    def train(self, states, action, action_prob_old, reward, states_next, done_masks):
        gaes, td_target = self.get_gaes(states, states_next, reward, done_masks)
        _, loss = tf.get_default_session().run([self.train_op, self.loss], feed_dict={self.X: states, self.actions: action, self.action_prob_old: action_prob_old, self.gaes: gaes, self.td_target: td_target})

        return loss

    def sample_action(self, X):
        return tf.get_default_session().run([self.action, self.action_prob, self.v_pred], feed_dict={self.X: X})

    def get_gaes(self, states, states_next, rewards, done_masks):
        rewards = np.array(rewards)
        v_pred = tf.get_default_session().run([self.v_pred], feed_dict={self.X: states})
        v_pred = np.array(v_pred).reshape(-1)
        v_pred_next = tf.get_default_session().run([self.v_pred], feed_dict={self.X: states_next})
        v_pred_next = np.array(v_pred_next).reshape(-1)
        td_target = rewards + self.gamma * v_pred_next * done_masks
        delta = td_target - v_pred
        gaes = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t
            gaes.append(advantage)
        gaes.reverse()

        return gaes, td_target

