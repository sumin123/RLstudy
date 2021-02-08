import gym
import tensorflow as tf
from net import net
import numpy as np

def main():
    env = gym.make('CartPole-v1')
    gamma = 0.99
    roll_out = 20
    score = 0.0
    print_interval = 20
    model = net()
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
                old_probs = []
                v_preds = []
                v_preds_next = []
                for t in range(roll_out):
                    a, a_prob, v_pred = model.sample_action(s.reshape(-1, 4))
                    s_prime,  r, done, info = env.step(a[0])
                    states.append(s)
                    actions.append(a)
                    rewards.append(r/100.0)
                    s_primes.append(s_prime)
                    old_probs.append(a_prob[0][a])
                    v_preds.append(v_pred)
                    s = s_prime

                    score += r
                    if done:
                        break
                for i in range(len(v_preds) - 1):
                    v_preds_next.append(float(v_preds[i+1][0]))
                v_preds_next.append(float(0))
                model.train(states, np.array(actions).reshape(-1), np.array(old_probs).reshape(-1), rewards, v_preds, v_preds_next)

            if n_epi % print_interval == 0 and n_epi != 0:
                print("episode: %s\tscore: %s" % (n_epi, score / print_interval))
                score = 0
if __name__ == '__main__':
    main()