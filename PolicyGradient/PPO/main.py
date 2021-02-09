import gym
import tensorflow as tf
from net import Net
import numpy as np

def main():
    env = gym.make('CartPole-v1')
    K = 4
    roll_out = 20
    score = 0.0
    print_interval = 20
    model = Net()
    is_render = False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n_epi in range(10000):
            s = env.reset()
            done = False
            while not done:
                states = []
                actions = []
                rewards = []
                s_primes = []
                prob_a = []
                done_masks = []
                for t in range(roll_out):
                    if is_render:
                        env.render()
                    a, a_prob, v_pred = model.sample_action(s.reshape(-1, 4))
                    s_prime,  r, done, info = env.step(a[0])
                    states.append(s)
                    actions.append(a)
                    rewards.append(r/100.0)
                    s_primes.append(s_prime)
                    prob_a.append(a_prob[0][a])
                    done_mask = 0.0 if done else 1.0
                    done_masks.append(done_mask)
                    s = s_prime

                    score += r
                    if done:
                        break

            for i in range(K):
                loss = model.train(states, np.array(actions).reshape(-1), np.array(prob_a).reshape(-1), rewards, s_primes, done_masks)

            if n_epi % print_interval == 0 and n_epi != 0:
                print("episode: %s\tscore: %s" % (n_epi, score / print_interval))
                if score > 200 * print_interval:
                    is_render = False
                score = 0


if __name__ == '__main__':
    main()