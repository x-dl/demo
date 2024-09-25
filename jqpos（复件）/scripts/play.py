from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from omegaconf import DictConfig, OmegaConf
import hydra


import sys
sys.path.append("..")
# from utils.NormalizeActionWrapper import NormalizeActionWrapper
from envs.cartpole.cartople import CartpoleRefEnv
import torch
# import gym
#
# gym.make("cartpole")

import numpy as np
import gym

class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        action_space = env.action_space
        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        self.low, self.high = action_space.low, action_space.high

        # 重塑动作空间范围
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self):
        """
        Reset the environment
        """
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # 重新把动作从[-1,1]放缩到原本的[low,high]
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        return obs, reward, done, info



class MyNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, model_path: str):
        super(MyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        model_a = PPO.load(model_path)
        # model_a = PPO.load("../../log/M04/2023-05-29/14-23-46/model_saved/ppo_M04_2150400.zip")
        params = model_a.policy.state_dict()
        # for name, value in params.items():
        #     print(name, value)
        self.fc1.weight.data = params["mlp_extractor.policy_net.0.weight"]
        self.fc1.bias.data = params["mlp_extractor.policy_net.0.bias"]
        self.fc2.weight.data = params["mlp_extractor.policy_net.2.weight"]
        self.fc2.bias.data = params["mlp_extractor.policy_net.2.bias"]
        self.fc3.weight.data = params["action_net.weight"]
        self.fc3.bias.data = params["action_net.bias"]
    def forward(self, obs_a: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs_a))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))




@hydra.main(version_base=None, config_path="../envs/cartpole", config_name="config")
def run_play(cfg: DictConfig) -> None:
    env = CartpoleRefEnv(cfg=cfg)
    env = NormalizeActionWrapper(env)
    # model = PPO.load("../log/cartpole1/2023-12-04/11-47-53/model_saved/ppo_M04_153600.zip", env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyNetwork(4, 1, 256, "../log/cartpole1/2023-12-22/15-25-04/model_saved/ppo_M04_204800.zip")
    model.to(device)
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    obs = env.reset()
    for i in range(10000):
        # print(action)
        obs = torch.from_numpy(obs).float().to(device)
        action = model.forward(obs)
        action = action.detach().cpu().numpy()
        obs, rewards, done, info = env.step(action)


if __name__ == '__main__':
    run_play()