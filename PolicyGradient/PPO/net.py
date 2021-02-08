import tensorflow as tf
import numpy as np

class net():
    def __init__(self):
        super(net, self).__init__()
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 2

        self.build_network('net')
        self.build_net_train('net')


    def build_network(self, name):
        ob_space = 4
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 4], name='X')

            with tf.variable_scope('policy_net'):
                layer1 = tf.layers.dense(inputs=self.X, units=256, activation=tf.nn.relu)
                self.action_prob = tf.layers.dense(inputs=layer1, units=2, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer1 = tf.layers.dense(inputs=self.X, units=256, activation=tf.nn.relu)
                self.v_pred = tf.layers.dense(inputs=layer1, units=1, activation=None)

            self.action = tf.argmax(self.action_prob, axis=1)
            self.action_prob_select = tf.reduce_max(self.action_prob, axis=1)

    def build_net_train(self, name):
        with tf.variable_scope('train_op'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='action')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
            self.action_prob_old = tf.placeholder(dtype=tf.float32, shape=[None], name='action_prob_old')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        with tf.variable_scope('loss_pi'):
            ratio = tf.exp(tf.log(self.action_prob_select) - tf.log(self.action_prob_old))
            clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratio), tf.multiply(self.gaes, clipped_ratio))
            loss_clip = tf.reduce_mean(loss_clip)

        with tf.variable_scope('loss_v'):
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None], name='td_target')
            loss_v = tf.squared_difference(self.td_target, self.v_pred)
            loss_v = tf.reduce_mean(loss_v)

        with tf.variable_scope('loss'):
            loss = loss_clip - 1 * loss_v
            loss = -loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.train_op = self.optimizer.minimize(loss, var_list=model_var)

    def train(self, states, action, action_prob_old, reward, v_pred, v_pred_next):
        gaes, td_target = self.get_gaes(reward, v_pred, v_pred_next)
        tf.get_default_session().run([self.train_op], feed_dict={self.X: states, self.actions: action, self.action_prob_old: action_prob_old, self.gaes: gaes, self.td_target: td_target})

    def sample_action(self, X):
        return tf.get_default_session().run([self.action, self.action_prob, self.v_pred], feed_dict={self.X: X})

    def get_gaes(self, rewards, v_pred, v_pred_next):
        rewards = np.array(rewards)
        v_pred = np.array(v_pred).reshape(-1)
        v_pred_next = np.array(v_pred_next)
        td_target = rewards + self.gamma * v_pred_next
        delta = td_target - v_pred
        gaes = []
        advantage = 0
        for delta_t in delta[::-1]:
            gaes.append(self.gamma * self.lmbda * advantage + delta_t)
        gaes.reverse()

        return gaes, td_target

