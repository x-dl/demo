# isolated cassie env
import math
from math import floor
from math import sqrt
import gym
import torch
from gym import spaces
import numpy as np
import os
import random
import copy
import pickle
import mujoco
import mujoco_viewer
import yaml
# from scipy.linalg import cho_factor, cho_solve
import csv

class CartpoleRefEnv(gym.Env):
    def __init__(self, cfg, **kwargs):
        self.config = cfg
        self.model = self.config['system']['root_path'] + self.config['system']['mjcf_path']
        self.visual = self.config['system']['visual']
        self.model = mujoco.MjModel.from_xml_path(self.model)
        self.data = mujoco.MjData(self.model)
        if self.visual:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.dynamics_randomization = self.config['system']['dynamics_randomization']
        self.termination = False

        # state buffer
        self.state_buffer = []
        self.buffer_size = self.config['env']['state_buffer_size']  # 1

        # Observation space and State space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.buffer_size * 4,))#目的地坐标+2
        self.action_high = np.array([0.39], dtype=np.float32)
        self.action_space = spaces.Box(0.01, self.action_high, dtype=np.float32)
        self.last_action = np.zeros(len(self.action_high), dtype=np.float32)

        self.torques = np.zeros(len(self.action_high), dtype=np.float32)

        self.P = [100]
        self.D = [5]

        self.joint_vel_left = np.zeros(4, dtype=np.float32)
        self.joint_vel_right = np.zeros(4, dtype=np.float32)

        self.simrate = self.config['control']['decimation']  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time = 0  # number of time steps in current episode
        self.counter = 0  # number of phase cycles completed in episode

        self.time_limit = self.config['env']['time_limit']
        self.time_buf = 0

        self.reward_buf = 0
        self.reward = 0
        self.reward_a = 0
        self.reward_a_buf = 0
        self.reward_b = 0
        self.reward_b_buf = 0
        self.reward_c = 0
        self.reward_c_buf = 0
        self.reward_d = 0
        self.reward_d_buf = 0

        self.damping_low = 0.3
        self.damping_high = 3.0
        self.mass_low = 0.8
        self.mass_high = 1.2
        self.fric_low = 0.5
        self.fric_high = 1.2
        self.com_low =0.8
        self.com_high = 1.2
        self.in_low = 0.5
        self.in_high = 2
        self.kds = 5
        self.kps = 80

        self.default_damping = self.model.dof_damping.copy()
        self.default_mass = self.model.body_mass.copy()
        self.default_ipos = self.model.body_ipos.copy()
        self.default_fric = self.model.geom_friction.copy()
        self.default_rgba = self.model.geom_rgba.copy()
        self.default_quat = self.model.geom_quat.copy()
        self.default_inertia = self.model.body_inertia.copy()
        self.history_qpos = np.array([0, 0], dtype=np.float32)
        # self.nnmodel = NeuralNetwork(1, 1)
        # self.nnmodel.load_state_dict(torch.load('saved_model.pth', map_location=torch.device('cpu')))
        self.f = open('actiona.csv', 'w', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.f)

    def set_const(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step_simulation(self, action):
        target = action
        # action = [0.1]
        self.joint_vel_left[0] = 0.7 * self.joint_vel_left[0] + 0.3 * self.data.sensordata[2]
        self.data.ctrl[0] = self.P[0] * (target[0] - self.data.qpos[0]) - self.D[0] * self.joint_vel_left[0]
        # if self.time % 80 == 0:
        #     self.data.xfrc_applied[3][2] = np.random.uniform(-0.3, 0.3)
        # self.data.ctrl = action
        self.torques = self.data.ctrl
        mujoco.mj_step(self.model, self.data)


    def step(self, action):
        # action = 0.7 * self.last_action + 0.3 * action
        for _ in range(60):
            self.step_simulation(action)
        self.csv_writer.writerow([action[0], self.data.qpos[0]])
        self.counter += 1
        self.time += 1
        # print(self.counter)
        obs = self.get_state()
        # print(obs)
        self.check_termination_criteria()
        if self.visual:
            self.render()
        reward = self.compute_reward(action)
        self.last_action = action
        return obs, reward, self.done, {}

    def check_termination_criteria(self):
        height = self.data.qpos[0]
        self.termination = height > 0.4 or height < 0.00 or np.abs(self.data.qpos[1]) >= 10 * np.pi
        self.done = self.termination or self.time >= self.time_limit


    def reset(self):
        if self.time != 0:
            self.reward_buf = self.reward
            self.reward_a_buf = self.reward_a/self.time
            self.reward_b_buf = self.reward_b / self.time
            self.reward_c_buf = self.reward_c / self.time
            self.reward_d_buf = self.reward_d / self.time
            self.time_buf = self.time

        if self.dynamics_randomization:
            damp = self.default_damping.copy()
            mass = self.default_mass.copy()
            inertia = self.default_inertia.copy()
            # 应用动力学随机化
            self.model.dof_damping[0] = np.random.uniform(damp[0] * self.damping_low, damp[0] * self.damping_high)
            self.model.dof_damping[1] = random.uniform(damp[1] * self.damping_low, damp[1] * self.damping_high)

            # self.model.body_mass[1] =random.uniform(mass[1] * self.mass_low, mass[1] * self.mass_high)
            self.model.body_mass[3] = np.random.uniform(mass[3] * self.mass_low, mass[3] * self.mass_high)
            self.model.body_inertia[3] = np.random.uniform(inertia[3] * self.in_low, inertia[3] * self.in_high)
            # self.model.geom_friction[1][1] = random.uniform (self.model.geom_friction[1][1] * self.fric_low, self.model.geom_friction[1][1] * self.fric_high)
            # self.model.geom_friction[2][0] = random.uniform (self.model.geom_friction[2][0] * self.fric_low, self.model.geom_friction[2][0] * self.fric_high)
            # self.data.subtree_com[2][2] = np.random.uniform(self.data.subtree_com[2][2] * self.com_low, self.data.subtree_com[2][2] * self.com_high )

        self.reward, self.reward_a,self.reward_b,self.reward_c,self.reward_d = 0,0,0,0,0
        self.time = 0
        self.termination = False
        self.set_const()
        return self.get_state()
    def get_state(self):
        self.qpos = np.copy(self.data.sensordata[0:2])
        self.qpos[1] = -(np.mod(self.qpos[1], 2*np.pi) - np.pi)
        self.qvel = np.copy(self.data.sensordata[2:4])
        # print(self.qpos)
        return np.concatenate([self.qpos[0:1], self.qvel[0:1], self.qpos[1:], self.qvel[1:]])

    def render(self):
        return self.viewer.render()

    def compute_reward(self, action):

        pos = self.data.sensordata[1]
        vel = self.data.sensordata[3]
        target_pos = 0  # 目标位置为3.14（弧度）
        target_vel = 0  # 目标速度为0
        target_sub = 0.12  #
        angle = target_pos - (-(np.mod(self.data.qpos[1], 2*np.pi) - np.pi))

        reward_a = 0.3 * np.exp(-50*((self.data.subtree_com[3][2] - 0.17) ** 2))
        # reward_a =  self.data.subtree_com[3][2]
        reward_b = 0.7 * np.exp(-40*(angle) ** 2)
        reward_c = 0
        if np.abs(self.data.sensordata[3]) > 14:
            reward_c = -0.05

        # reward_c += 0.05 * np.exp(-40 * (action[0] - self.last_action[0]) ** 2)
        reward_c +=  -0.0 * np.sum(np.square(self.last_action - action))
        reward_c += -0.0000 * np.sum(np.square(self.torques))
        reward_c += -0.000 * np.sum(np.square(self.data.qvel[6:12]))
        reward_c += -0.0000 * np.sum(np.square(self.data.qacc[6:12]))
        reward_d = 0.0 * np.exp(-40 * (self.data.sensordata[0] - 0.2) ** 2)

        ######################### 课程学习 ####################################
        # if self.counter < 800000:
        #     reward = 0.6 * reward_a + 0.3 * 0.05 * reward_b + 0.05 * reward_c
        # elif 800000 < self.counter :
        reward = reward_a + reward_b + reward_c + reward_d

        # if self.data.sensordata[0] < 0.1:
        #     reward_d += -0.1 * np.exp(0.1 - self.data.qpos[0])
        # elif 0.1 < self.data.qpos[0] < 0.3:
        #     reward_d += 0
        # elif self.data.sensordata[0] > 0.3:
        #     reward_d += -0.1 * np.exp(self.data.qpos[0] - 0.3)

        # reward_b = -np.cos(self.data.sensordata[1]) - 0.5*(self.data.sensordata[0] - 0.2)**2
        # body_a = self.data.subtree_com[2][0]
        # body_b = self.data.subtree_com[3][0]
        # reward_c = -0 * np.sum(np.square(body_b - body_a)) + reward_a
        # pos = self.data.qpos[1]
        # vel = self.data.qvel[1]
        # target = 3.14
        # angle = np.abs(pos - target)
        # reward_d = np.exp(-10 * angle)
        # reward_b = -0.05 * np.sum(np.square(self.last_action - action))
        # reward = reward_a + reward_b + reward_c +  reward_d
        self.reward_a += reward_a
        self.reward_b += reward_b
        self.reward_c += reward_c
        self.reward_d += reward_d
        self.reward += reward
        return reward



if __name__ == "__main__":
    # 导入动作规范化wrapper
    import sys
    sys.path.append("../..")
    from utils.NormalizeActionWrapper import NormalizeActionWrapper

    with open('./config.yaml', 'rb') as stream:
        config = yaml.safe_load(stream)
    config['system']['root_path'] = '../..'
    config['system']['visual'] = True

    env = CartpoleRefEnv(cfg=config)
    print(env.action_space.high)
    print(env.action_space.low)

    env = NormalizeActionWrapper(env)
    obs = env.reset()
    print(env.action_space)

    for i in range(10000):
        # action = np.zeros(8)
        action = np.random.random(1)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()


def compute_reward(self, action):
    custom_footheight = np.array(self.custom_footheight())
    real_footheight = np.array([self.foot_pos[2], self.foot_pos[5]])
    ref_penalty = np.sum(np.square(custom_footheight - real_footheight))
    ref_penalty = ref_penalty / 0.0025
    orientation_penalty = (self.data.sensordata[25]) ** 2 + (self.data.sensordata[26]) ** 2 + (
    self.data.sensordata[27]) ** 2
    orientation_penalty = orientation_penalty / 0.1

    vel_penalty = (self.speed - self.data.sensordata[28]) ** 2 + (self.side_speed - self.data.sensordata[29]) ** 2 + (
    self.data.sensordata[30]) ** 2

    vel_penalty = vel_penalty / max(0.5 * (self.speed * self.speed + self.side_speed * self.side_speed), 0.01)

    rew_action_rate = -0.01 * np.sum(np.square(self.last_action - action))
    rew_torque = -0.00001 * np.sum(np.square(self.torques))
    rew_torque += -0.0002 * np.sum(np.square(self.data.qvel[6:12]))

    rew_torque += -0.00001 * np.sum(np.square(self.data.qacc[6:12]))

    rew_ref = 0.3 * np.exp(-ref_penalty)

    l_f_h = self.data.xpos[14][2]
    l_b_h = self.data.xpos[16][2]
    r_f_h = self.data.xpos[21][2]
    r_b_h = self.data.xpos[23][2]
    rew_a = - 0.0 * (l_f_h - l_b_h) ** 2
    rew_b = - 0.0 * (r_f_h - r_b_h) ** 2

    # 防止身體晃動
    base_pos = self.data.xpos[1][1]
    rew_upper = 0.0 * np.exp(-40 * (base_pos) ** 2) + rew_a + rew_b

    rew_spring = 0  # .1 * np.exp(-spring_penalty)
    rew_ori = 0.125 * np.exp(-orientation_penalty)
    rew_vel = 0.375 * np.exp(-vel_penalty)  #
    rew_termin = -10 * self.termination

    R_star = 1
    Rp = (0.75 * np.exp(-vel_penalty) + 0.25 * np.exp(-orientation_penalty)) / R_star

    Ri = np.exp(-ref_penalty) / R_star
    Ri = (Ri - 0.4) / (1.0 - 0.4)

    omega = 0.4

    reward = (1 - omega) * Ri + omega * Rp + rew_spring + rew_termin + rew_action_rate + rew_torque + rew_upper
