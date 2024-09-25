import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from omegaconf import DictConfig, OmegaConf
import hydra

import sys
sys.path.append("../..")
from utils.NormalizeActionWrapper import NormalizeActionWrapper
from envs.cartpole.cartople import CartpoleRefEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def make_env(env_id, cfg):
    def _f():
        if env_id == 0:
            env = CartpoleRefEnv(cfg=cfg, visual=False)
            env = NormalizeActionWrapper(env)
        else:
            env = CartpoleRefEnv(cfg=cfg,visual=False)
            env = NormalizeActionWrapper(env)
        return env
    return _f





@hydra.main(version_base=None, config_path="../envs/cartpole", config_name="config")
def run_train(cfg: DictConfig) -> None:
    cfg.system.visual = False
    # get log path name
    log_dir = cfg['system']['log_path']['dir']
    # make env
    envs = [make_env(seed, cfg) for seed in range(cfg['env']['num_envs'])]
    envs = SubprocVecEnv(envs)
    # define callback function
    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, log_dir, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            self.log_dir = log_dir

        def _on_step(self) -> bool:
            #self.logger.record('reward/ref', np.mean(self.training_env.get_attr('rew_ref_buf')))
            self.logger.record('reward/spring', np.mean(self.training_env.get_attr('reward_buf')))
            self.logger.record('reward/a', np.mean(self.training_env.get_attr('reward_a_buf')))
            self.logger.record('reward/b', np.mean(self.training_env.get_attr('reward_b_buf')))
            self.logger.record('reward/c', np.mean(self.training_env.get_attr('reward_c_buf')))
            self.logger.record('reward/d', np.mean(self.training_env.get_attr('reward_d_buf')))
            if self.n_calls % 51200 == 0:
                print("Saving new best model")
                self.model.save(self.log_dir + f"/model_saved/ppo_M04_{self.n_calls}")

            return True



    # make policy rule
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=[dict(pi=[int(i) for i in cfg['trainer']['pi_net_arch']],
                                        vf=[int(i) for i in cfg['trainer']['vf_net_arch']])])



    model = PPO(cfg['trainer']['policy'], envs, verbose=1, n_steps=cfg['trainer']['n_steps'],
                      policy_kwargs=policy_kwargs, batch_size=cfg['trainer']['batch_size'],
                      tensorboard_log=log_dir, device=cfg['trainer']['device'], use_sde=False)
    model.is_tb_set = False

    model.learn(total_timesteps=int(4e7), n_eval_episodes=cfg['trainer']['n_eval_episodes'],
                callback=TensorboardCallback(log_dir=log_dir))
    # model.save("ppo_cassie")


if __name__ == '__main__':
    run_train()



