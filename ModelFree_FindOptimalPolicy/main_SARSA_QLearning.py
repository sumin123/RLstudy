from GridWorld import *
from QAgent_SARSA import *
from QAgent_Q_Learning import *


def main():
    env = GridWorld()
    #agent = QAgent_SARSA()
    agent = QAgent_Q_Learning()

    for n_epi in range(5000):
        done = False
        s = env.reset()
        while not done:
            action = agent.select_action(s)
            s_prime, reward, done = env.step(action)
            transition = [s, action, reward, s_prime]
            agent.update_table(transition)
            s = s_prime
        agent.anneal_eps()
    agent.show_table()

if __name__ == '__main__':
    main()