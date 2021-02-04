from Agent import *
from GridWorld import *

def main():
    gamma = 1.0
    alpha = 0.01
    epoch = 50000

    env = GridWorld()
    agent = Agent()
    v = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(epoch):

        done = False
        while not done:
            old_x, old_y = env.get_state()
            action = agent.select_action()
            (new_x, new_y), reward, done = env.step(action)
            new_x, new_y = env.get_state()
            #v[old_x][old_y] += (reward + v[new_x][new_y] * gamma - v[old_x][old_y]) * alpha
            v[old_x][old_y] = v[old_x][old_y] * (1 - alpha) + alpha * (reward + gamma * v[new_x][new_y])
        env.reset()

    for i in range(4):
        print(v[i])

if __name__ == '__main__':
    main()