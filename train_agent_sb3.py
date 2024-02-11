from datetime import datetime

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from kart_env import MarioKartEnvMultiDiscrete, MarioKartEnv
from utils import MarioAtariWrapper, TensorboardCallback

if __name__ == '__main__':

    now = datetime.now()

    def make_env_with_stats(include_lower_frame=True, multi_discrete=False):
        if multi_discrete:
            env = MarioKartEnvMultiDiscrete(include_lower_frame=include_lower_frame)
        else:
            env = MarioKartEnv(include_lower_frame=include_lower_frame)

        return env

    env = make_vec_env(make_env_with_stats, n_envs=12, seed=np.random.randint(0, 2**31 -1), # need to specify that the dtype is int64 so it works on windows
                         env_kwargs={'include_lower_frame': True, 'multi_discrete': False},
                         vec_env_cls=SubprocVecEnv,
                         wrapper_class=MarioAtariWrapper,
                         monitor_kwargs={'info_keywords': ('percent_complete', 'place')})
    env = VecFrameStack(env, n_stack=4)

    # get the current datetime so we can use it to name our tensorboard log directory
    output_folder = f'./runs/{now.strftime("%Y-%m-%d_%H-%M-%S")}/'

    model = RecurrentPPO('CnnLstmPolicy', env, verbose=1, tensorboard_log=output_folder, policy_kwargs={'enable_critic_lstm': False}, n_steps=2048)

    #model = RecurrentPPO.load('./runs/2023-10-22_08-40-18/mario-kart-rppo.zip', tensorboard_log=output_folder, env=env) # okay, maybe try startin with a pretrained model

    model.learn(total_timesteps=480_000, reset_num_timesteps=True, callback=TensorboardCallback())

    # now that we have a trained model, we can save it and load it later
    model.save(output_folder + 'mario-kart-rppo')