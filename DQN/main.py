import gym
from Qnet import *
from ReplayBuffer import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#hyperparameter
LEARNING_RATE = 0.0001
BUFFER_LIMIT = 50000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 10


def get_copy_var_ops(target_scope_name, src_scope_name):
    # Copy variables src_scope to target_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

    for src_var, target_var in zip(src_vars, target_vars):
        op_holder.append(target_var.assign(src_var.value()))

    return op_holder


def main():
    buffer = ReplayBuffer(BUFFER_LIMIT)
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print_interval = 20
    is_render = False

    with tf.Session() as sess:
        net = Qnet(sess, input_dim, action_dim, LEARNING_RATE, name="main")
        target_net = Qnet(sess, input_dim, action_dim, LEARNING_RATE, name="target")
        sess.run(tf.global_variables_initializer())
        copy_ops = get_copy_var_ops(target_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        loss = 0.0
        score = 0.0
        for n_epi in range(10000):
            s = env.reset()
            done = False
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
            while not done:
                if is_render:
                    env.render()
                a = np.argmax(net.sample_action(s.reshape(1, 4)))
                s_prime, r, done, _ = env.step(a)
                score += r
                done_mask = 0.0 if done else 1.0
                transition = [s, a, r/100.0, s_prime, done_mask]
                buffer.put(transition)
                if done:
                    break
                s = s_prime

            if buffer.size() > 2000:
                s, a, r, s_prime, done_mask = buffer.sample(BATCH_SIZE)
                loss, _ = net.update(target_net, s, a, r, s_prime, done_mask)
            if n_epi % TARGET_UPDATE_FREQ == 0:
                sess.run(copy_ops)
            if n_epi % print_interval == 0 and n_epi != 0:
                print("n_episode :{}, score : {:.1f}, loss: {}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval, loss, buffer.size(), epsilon*100))
                if score > 200 * print_interval:
                    is_render = True
                score = 0.0


if __name__ == '__main__':
    main()

