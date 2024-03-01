import numpy


class ENV:
    def __init__(self):
        self.actionspace = [0, 1, 2, 3]
        self.n_action = 4
        self.reward_range = (-float('inf'), float('inf'))
        self.work = numpy.random.randint(0, 4, 1)
        self.edgecpu = [0.2, 0.4, 0.8, 1.0]
    def reset(self):
        self.work = numpy.random.randint(0, 4, 1)
        observation = self.work
        return observation

    def step(self, action):
        reward = -(self.work/self.edgecpu[action])
        work_ = numpy.random.randint(0, 4, 1)
        done = True
        info = 0
        return work_, reward, done, info

# 为线性的选择避免一直选择的都是同一个服务器，需考虑时延敏感、能耗敏感等敏感度
