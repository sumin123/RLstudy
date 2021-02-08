import collections
import random

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

    def size(self):
        return len(self.buffer)