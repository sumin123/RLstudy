from Agent import *
from GridWorld import *

def main():
    env = GridWorld()
    agent = Agent()
    v = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    gamma = 1.0
    alpha = 0.0001
    for i in range(50000):
        traj =[]
        done = False
        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            traj.append([x, y, reward])
        env.reset()
        g = 0
        for i in traj[::-1]:
            x, y = i[0], i[1]
            v[x][y] = v[x][y] * (1 - alpha) + g * alpha
            g = i[2] + gamma * g
    for i in range(4):
        print(v[i])

if __name__ == '__main__':
    main()