import gym
from Policy import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
LEARNING_RATE = 0.0002
GAMMA = 0.98

def main():
    env = gym.make('CartPole-v0')
    pi = Policy(LEARNING_RATE, GAMMA)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score: {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()