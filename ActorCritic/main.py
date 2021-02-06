import gym
import torch
from torch.distributions import Categorical
from ActorCritic import ActorCritic

LEARNING_RATE = 0.0002
GAMMA = 0.98
n_rollout = 10


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic(LEARNING_RATE, GAMMA)
    score = 0.0
    print_interval = 20
    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            for i in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()

                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r
                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score: {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()


if __name__=='__main__':
    main()