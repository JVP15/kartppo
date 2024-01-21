import ray
import gymnasium as gym
from utils import MarioAtariWrapper
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from kart_env import MarioKartEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from ray import air
from ray import tune
import os

def create_wrapped_mario_kart_env(env_config):
    env = MarioKartEnv(include_lower_frame=env_config['include_lower_frame'])
    env = MarioAtariWrapper(env, screen_size=84, frame_skip=4)
    #env = FrameStack(env, num_stack=4)

    return env

if __name__ == '__main__':

    register_env('MarioKartEnv-Wrapped', create_wrapped_mario_kart_env)
    ray.init(num_gpus=1)


    config = PPOConfig()
    config = config.rollouts(num_rollout_workers=12)
    config = config.resources(num_gpus=1, num_gpus_per_learner_worker=1)
    config = config.environment('MarioKartEnv-Wrapped', env_config={'include_lower_frame':True})
    config = config.training(
        model={"use_lstm": True,"lstm_cell_size": 256},
        lr=0.0003,
        clip_param=0.2,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
        gamma=.99,
        lambda_=0.95,
        train_batch_size=2048
    )

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(stop={"training_iteration": 480_000}, storage_path=os.path.join(os.getcwd(),'runs/ray_results')),
        param_space=config.to_dict(),
    ).fit()
