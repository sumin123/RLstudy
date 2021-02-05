import gym
from Qnet import *
from ReplayBuffer import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#hyperparameter
LEARNING_RATE = 0.0005
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
        q = Qnet(sess, input_dim, action_dim, LEARNING_RATE, name="main")
        q_target = Qnet(sess, input_dim, action_dim, LEARNING_RATE, name="target")
        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(target_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        for n_epi in range(10000):
            s = env.reset()
            done = False
            score = 0.0
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))
            while not done:
                if is_render:
                    env.render()
                a = np.argmax(q.sample_action(s.reshape(-1, 4)))
                s_prime, r, done, _ = env.step(a)
                score += r
                done_mask = 0.0 if done else 1.0
                buffer.put([s, a, r, s_prime, done_mask])
                if done:
                    break
            if buffer.size() > 2000:
                s, a, r, s_prime, done_mask = buffer.sample(BATCH_SIZE)
                q.update(s, a, r, s_prime, done_mask)
            if n_epi % TARGET_UPDATE_FREQ == 0:
                sess.run(copy_ops)
            if n_epi % print_interval == 0 and n_epi != 0:
                print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval, buffer.size(), epsilon*100))
                if score > 200 * print_interval:
                    is_render = True


if __name__ == '__main__':
    main()

