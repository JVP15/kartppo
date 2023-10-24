import ray
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from kart_env import MarioKartEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

def create_wrapped_mario_kart_env(env_config):
    env = MarioKartEnv(include_lower_frame=env_config['include_lower_frame'])
    env = AtariWrapper(env, screen_size=84, frame_skip=4, terminal_on_life_loss=False, clip_reward=False)
    #env = FrameStack(env, num_stack=4)

    return env

if __name__ == '__main__':

    register_env('MarioKartEnv-Wrapped', create_wrapped_mario_kart_env)

    config = PPOConfig()
    config = config.rollouts(num_rollout_workers=12)
    config = config.resources(num_gpus=1)
    config = config.environment('MarioKartEnv-Wrapped', env_config={'include_lower_frame':True})
    config = config.training(model={"use_lstm": True,"lstm_cell_size": 256})
    algo = config.build()


    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

        if i % 5 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")

