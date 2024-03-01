import numpy as np


class ENV:
    def __init__(self):
        self.actionspace = [0, 1, 2, 3]
        self.n_action = 4
        self.reward_range = (-float('inf'), float('inf'))

        self.device_num = 1
        self.edge_num = 2

        #有多少个任务
        self.service_num = 4

        # 有多少任务liang
        self.data_list = np.random.uniform(2, 5, size=(1))

        # 每bit需要的clock
        self.per_bit_needed_clock = np.random.randint(1, 10, size=(1))

        # 排队时间,下一步再考虑
        self.cloud_wait = np.zeros(1)
        self.edge_wait = np.zeros(self.edge_num)
        self.device_wait = np.zeros(1)

        # 计算能力
        self.device_calp = np.random.uniform(1, 10, size=(self.device_num))
        self.edge_calp = np.random.uniform(20, 40, size=(self.edge_num))
        self.cloud_calp = np.array([np.random.uniform(50, 150)])

        # 服务在所需的最小计算能力，如果不足，则返回惩罚，对应一个
        self.service_need_calc = np.random.uniform(1, 10, size=(self.service_num))
        # 对应服务最大等待时间，需要考虑传输时间和排队时间和计算时间。
        # service_wait = trans time + wait time + calculate time

        # tans_v
        self.trans_to_edge = np.random.uniform(500, 1000, size=(self.edge_num))
        self.trans_to_cloud = np.array([np.random.uniform(500, 1000)])
        self.trans_power = np.random.uniform(10, 20, size=(self.device_num))

        self.device_hardware = np.random.uniform(0, 0.0005, size=(self.service_num))
        self.edge_hardware = np.random.uniform(0.0005, 0.001, size=(self.edge_num))
        self.cloud_hardware = np.array([np.random.uniform(0.001, 0.002)])

    def make(self):
        # self.env_state = np.concatenate([self.data_list, self.per_bit_needed_clock,
        #                                  self.service_need_calc, self.device_calp, self.edge_calp, self.device_calp,
        #                                  self.edge_calp,
        #                                  self.cloud_calp, self.cloud_wait, self.edge_wait, self.device_wait,
        #                                  self.trans_to_edge, self.trans_to_cloud,
        #                                  self.trans_power, self.device_hardware, self.edge_hardware, self.edge_hardware,
        #                                  self.cloud_hardware])
        self.env_state = np.concatenate([
                                         (self.data_list-2)/3, (self.per_bit_needed_clock-1)/9,
                                         (self.service_need_calc-1)/9,
                                         (self.device_calp-1)/9, (self.edge_calp-20)/20, (self.cloud_calp-50)/100,
                                         self.cloud_wait, self.edge_wait, self.device_wait,
                                         (self.trans_to_edge-500)/1000, (self.trans_to_cloud-500)/1000,
                                         (self.trans_power-10)/20, (self.device_hardware-0)/0.0005, (self.edge_hardware-0.0005)/0.0005,
                                         (self.cloud_hardware-0.001)/0.001,
                                         ])
        return self.env_state

    def reset(self):
        self.data_list = np.random.uniform(2, 5, size=(1))
        # self.device_calp = np.random.uniform(1, 10, size=self.device_num)
        # self.edge_calp = np.random.uniform(20, 40, size=self.edge_num)
        # self.cloud_calp = np.array([np.random.uniform(50, 150)])
        self.device_wait[0] = 0
        self.cloud_wait[0] = 0
        self.edge_wait[:] = 0
        state = self.make()
        return state

    def step(self, action, time):
        i = 0
        if action == 0:
            exec_time = self.data_list[i] * self.per_bit_needed_clock[i] / self.device_calp[0]
            e_cost = self.device_hardware[0] * self.device_calp[0] ** 3 * exec_time
            cost = exec_time + e_cost
            reward = -cost
            # self.device_wait[0] += exec_time
            #self.device_calp -= self.service_need_calc[i]
        elif action in range(1, self.edge_num + 1):
            # print(i)
            # print(self.data_list)
            # print(self.per_bit_needed_clock)
            # print(self.edge_calp)
            exec_time = self.data_list[i] * self.per_bit_needed_clock[i] / self.edge_calp[action - 1]
            tran_time = self.data_list[i] / self.trans_to_edge[action - 1]
            e_cost = self.edge_hardware[action - 1] * self.edge_calp[
                action - 1] ** 3 * exec_time + self.trans_power * tran_time
            cost = self.edge_wait[action - 1] + exec_time + tran_time + e_cost
            reward = -cost
            #self.edge_wait[action - 1] += exec_time
            #self.edge_calp[action - 1] -= self.service_need_calc[i]
        elif action == self.edge_num + 1:
            exec_time = self.data_list[i] * self.per_bit_needed_clock[i] / self.cloud_calp[0]
            tran_time = self.data_list[i] / self.trans_to_cloud
            e_cost = self.cloud_hardware[0] * self.cloud_calp ** 3 * exec_time + self.trans_power[0] * tran_time
            cost = self.cloud_wait + exec_time + tran_time + e_cost
            reward = -cost
            #self.cloud_wait[0] += exec_time
            #self.cloud_calp -= self.service_need_calc[i]
        else:
            reward = 0
        self.next_state = self.make()
        if time == self.service_num - 1:
            self.done = True
        else:
            self.done = False
        info = 0
        #print(reward)
        if reward > 0:
            a = 1
        return self.next_state, reward, self.done, info

