from GridWorld import *
from QAgent_MonteCarloControl import *

def main():
    env = GridWorld()
    agent = QAgent_MC()

    for n_epi in range(1000):
        done = False
        history = []
        s = env.reset()
        while not done:
            action = agent.select_action(s)
            s_prime, reward, done = env.step(action)
            history.append([s, action, reward, s_prime])
            s = s_prime
        agent.update_table(history)
        agent.anneal_eps()
    agent.show_table()

if __name__ == '__main__':
    main()