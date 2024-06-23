import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


from kart_env import MarioKartEnv, MarioKartEnvMultiDiscrete
from utils import MarioAtariWrapper
import time


if __name__ == '__main__':
    env = make_vec_env(MarioKartEnv, n_envs=1, #seed=np.random.randint(0, 2**31 -1), # need to specify that the dtype is int64 so it works on windows
                         env_kwargs={'include_lower_frame': True,},
                         vec_env_cls=SubprocVecEnv,
                         wrapper_class=MarioAtariWrapper,
                         monitor_kwargs={'info_keywords': ('percent_complete', 'place')})
    env = VecFrameStack(env, n_stack=4)

    action_space = env.action_space

    # load the model mario-kart-ppo.zip
    #model = RecurrentPPO.load('runs/2023-04-20_18-47-58/mario-kart-rppo.zip')
    model = RecurrentPPO.load('runs/2023-10-28_15-54-17/mario-kart-rppo')
    #model = RecurrentPPO.load('runs/2023-07-25_08-21-56/mario-kart-rppo')


    # run the model in the environment

    obs = env.reset()
    _states = None
    dones = np.ones((1,))
    episode_reward = 0
    timestep = 0
    while True:
        action, _states = model.predict(obs, state=_states, episode_start=dones, deterministic=True)

        # if timestep < 30:
        #     obs, rewards, dones, info = env.step([action_space.sample()])
        # else:
        obs, rewards, dones, info = env.step(action)

        episode_reward += rewards[0]
        print(f'Episode reward: {episode_reward}', '| Timestep: ', timestep)
        #print('Current checkpoint = ', info[0]['checkpoint'], 'Last checkpoint = ', info[0]['last_checkpoint'])
        #print('Current lap = ', info[0]['laps'])
        print('Percent Complete: ', info[0]['percent_complete'])
        print('Place = ', info[0]['place'])
        env.render(mode='human')
        #time.sleep(0.05)
        if dones[0]:
            #obs = env.reset()
            episode_reward = 0

        timestep +=1
