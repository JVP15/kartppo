import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import cv2
from gymnasium.wrappers import FrameStack

from utils import TensorboardCallback, MarioAtariWrapper


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import RecurrentPPO

from desmume.emulator import DeSmuME, SCREEN_PIXEL_SIZE, SCREEN_PIXEL_SIZE_BOTH, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH
from desmume.controls import Keys, keymask

TOTAL_LAPS = 3


# addresses from https://tasvideos.org/GameResources/DS/MarioKartDS
PLAYER_DATA_ADDR = 0x217ACF8
SPEED_OFFSET = 0x2A8
TOP_SPEED_OFFSET = 0xD0
AIR_OFFSET = 0x3DD
PLACE_OFFSET = 0x3D8

CHECKPOINT_DATA_ADDR = 0x021661B0
CHECKPOINT_OFFSET = 0xDAE

IDX_DRIVE = 0
ACTION_NOOP = 0
ACTION_ACCELERATE = 1
ACTION_ACCELERATE_LEFT = 2
ACTION_ACCELERATE_RIGHT = 3
ACTION_ACCELERATE_ITEM = 4

IDX_ITEM = 1
ACTION_ITEM = 1


FRAMES_BEFORE_START = 256 # there is the 3, 2, 1 countdown, although 256 isn't the exact value, it's close enough.
TIMEOUT = 600 # if we don't pass a checkpoint after 600 frames (~10 seconds @ Mario Kart's naitive 60fps) then we terminate the episode

ROM_FILE = 'Mario Kart DS (USA) (En,Fr,De,Es,It)/Mario Kart DS (USA) (En,Fr,De,Es,It).nds'
#SAVESTATE_FILES = ('figure_8_grand_prix.dsv', )
#SAVESTATE_FILES = ('figure_8_time_trial.dsv', )
SAVESTATE_FILES = ('figure_8_time_trial.dsv', 'yoshi_falls_time_trial.dsv',)
#SAVESTATE_FILES = ('luigis_mansion_time_trial.dsv', )
#SAVESTATE_FILES = ('figure_8_time_trial.dsv', 'yoshi_falls_time_trial.dsv', 'cheep_cheep_beach_time_trial.dsv','rainbow_road_time_trial.dsv', )

class MarioKartEnv(gym.Env):

    def __init__(self, include_lower_frame=False, rom_file=ROM_FILE, savestate_files=SAVESTATE_FILES):
        super().__init__()

        # create a spec for the environment
        #self.spec = gym.envs.registration.EnvSpec('MarioKartDS-v0-NoFrameskip')
        self.ale = None

        self.include_lower_frame = include_lower_frame

        self.action_space = gym.spaces.Discrete(5) # no op, accelerate, accelerate + left, accelerate + right

        if self.include_lower_frame:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 3), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

        self.emu = DeSmuME()
        self.emu.volume_set(0) # emulator makes noise even when headless

        self.mem = self.emu.memory

        self.emu.open(rom_file)
        self.savestate_files = savestate_files
        self.grand_prix = False

        self.checkpoint = None
        self.final_checkpoint = None # this is useful for detecting when we've finished a lap; we set this during reset()
        self.frames_since_checkpoint = -FRAMES_BEFORE_START # if we don't make any forward progress, we're stuck and we terminate the episode (also give a buffer for the countdown)

        self.top_speed = 0

        self.laps = -1  # we add 1 to the lap counter whenever we go from the last checkpoint to checkpoint 0, and we start in the last checkpoint, so it's easier to just start at -1

        self.render_mode = 'human'

    def get_action_meanings(self):
        return ['NOOP', 'ACCELERATE', 'ACCELERATE_ITEM', 'ACCELERATE_LEFT', 'ACCELERATE_RIGHT']

    def _get_checkpoint(self):
        checkpoint_ptr = self.mem.signed.read_long(CHECKPOINT_DATA_ADDR)
        checkpoint = self.mem.unsigned.read_byte(checkpoint_ptr + CHECKPOINT_OFFSET)

        return checkpoint

    def _get_obs(self):
        screen_buffer = self.emu.display_buffer_as_rgbx()
        screen_pixels = np.frombuffer(screen_buffer, dtype=np.uint8)

        if self.include_lower_frame:
            screen = screen_pixels[:SCREEN_PIXEL_SIZE_BOTH * 4] # see https://py-desmume.readthedocs.io/en/latest/quick_start.html#custom-drawing
            screen = screen.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))[..., :3] # drop the alpha channel
        else:
            screen = screen_pixels[:SCREEN_PIXEL_SIZE * 4]
            screen = screen.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 4))[..., :3]

        return screen

    def _act(self, action):
        # it's just easier to clear all of the keys first and then set the ones we want instead of trying to figure out which ones are already set
        self.emu.input.keypad_rm_key(Keys.NO_KEY_SET)

        if action == ACTION_ACCELERATE:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
        elif action == ACTION_ACCELERATE_ITEM:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
            self.emu.input.keypad_add_key(keymask(Keys.KEY_X))
        elif action == ACTION_ACCELERATE_LEFT:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
            self.emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
        elif action == ACTION_ACCELERATE_RIGHT:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
            self.emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

    def _load_savestate(self):
        savestate_file = np.random.choice(self.savestate_files)

        self.grand_prix = 'grand_prix' in savestate_file

        self.emu.savestate.load_file(savestate_file)

    def step(self, action):
        self._act(action)

        self.emu.cycle()

        cur_checkpoint = self._get_checkpoint()

        reward = -1/60 if self.frames_since_checkpoint >= 0 else 0 # lose 1 point per second after the race starts (we start at -256 frames to deal with the countdown, so we can use it here too)
        terminated = False
        truncated = False

        player_data_ptr = self.mem.signed.read_long(PLAYER_DATA_ADDR)
        place = self.mem.unsigned.read_byte(player_data_ptr + PLACE_OFFSET) if self.grand_prix else 9 #
        in_air = self.mem.unsigned.read_byte(player_data_ptr + AIR_OFFSET)
        speed = self.mem.unsigned.read_long(player_data_ptr + SPEED_OFFSET)

        #if speed == 0 and in_air:
        #    reward -= 100/60 # lose 100 points per second while we are being held by lakitu

        speed_penalty = (speed - self.top_speed) / self.top_speed * 10 / 60 # lose up to 10 points per second if the kart is at 0 speed (it also gains points if it has a boost); also catches times when the max speed is unavoidably lower, but it is worthwhile
        speed_penalty = speed_penalty if self.frames_since_checkpoint >= 0 else 0 # don't penalize speed during the countdown

        reward += speed_penalty

        # print('top speed = ', self.top_speed, 'speed = ', speed, 'speed penalty = ', speed_penalty)
        # print('saved checkpoint = ', self.checkpoint, 'current checkpoint = ', cur_checkpoint)

        if (cur_checkpoint - self.checkpoint == 1   # if we've moved to the next checkpoint...
                or (cur_checkpoint == 0 and self.checkpoint == self.final_checkpoint)): # or if we've finished a lap, give a reward
            # completing a whole lap is worth 100 points, so we divide by the number of checkpoints to get the reward per checkpoint
            # checkpoints range from [0, last_checkpoint], so normally we'd add 1 to the denominator, but we give a reward for passing the finish line, so we don't need to.
            reward = 200 / (self.final_checkpoint)

            if cur_checkpoint == 0 and self.checkpoint == self.final_checkpoint: # once we've finished a lap increase the lap counter and give a larger reward
                self.laps += 1
                reward = 100

            self.checkpoint = cur_checkpoint
            self.frames_since_checkpoint = 0
        elif cur_checkpoint - self.checkpoint <= -1:
            reward -= 2 * 200 / (self.final_checkpoint) # if we go backwards, I'll give it a penalty that is twice as large as the reward for going forwards
            self.checkpoint = cur_checkpoint

        if not (in_air and speed == 0): # this checks if we're being respawned by lakitu, if so, we don't want to count it towards the timeout
            self.frames_since_checkpoint += 1

        if self.laps == TOTAL_LAPS:
            terminated = True

            if self.grand_prix:
                reward += 1000 / place

        elif self.frames_since_checkpoint > TIMEOUT:
            truncated = True
            place = 9 # if we timeout, even in grand prix, always place 9th (especially since places go 1-8 otherwise)

        percent_complete = self.laps / TOTAL_LAPS + self.checkpoint / (self.final_checkpoint + 1) / TOTAL_LAPS

        return self._get_obs(), reward, terminated, truncated, {'percent_complete': percent_complete, 'place': place}

    def render(self, mode='human'):
        if self.render_mode == 'human':
            frame = self._get_screen_rgb()
            cv2.imshow('Mario Kart', frame)
            cv2.waitKey(1)
        elif self.render_mode == 'rgb_array':
            return self._get_screen_rgb()
        else:
            return self._get_obs()

    def _get_screen_rgb(self):
        frame = self._get_obs()
        # resize the frame to make it 2x bigger
        if self.include_lower_frame:
            frame = cv2.resize(frame, (SCREEN_WIDTH * 2, SCREEN_HEIGHT_BOTH * 2), interpolation=cv2.INTER_NEAREST)
        else:
            frame = cv2.resize(frame, (SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2), interpolation=cv2.INTER_NEAREST)

        return frame

    def reset(self, seed=None, options=None):
        # NOTE: when we reset, we have to start at a time when the track is already loaded so we can get our current checkpoint (needed so that we can find the number of checkpoints)
        super().reset(seed=seed, options=options)

        self._load_savestate()
        self.checkpoint = self._get_checkpoint()
        self.final_checkpoint = self.checkpoint
        self.frames_since_checkpoint = -FRAMES_BEFORE_START
        self.laps = -1

        # the top speed at the start is the normal speed on the ground, so we can use that as a baseline
        player_data_ptr = self.mem.signed.read_long(PLAYER_DATA_ADDR)
        self.top_speed = self.mem.unsigned.read_long(player_data_ptr + TOP_SPEED_OFFSET)

        return self._get_obs(), {}


class MarioKartEnvMultiDiscrete(MarioKartEnv):
    """
    Mario Kart environment with a MultiDiscrete action space.
    """
    def __init__(self, include_lower_frame=False):
        super().__init__(include_lower_frame=include_lower_frame)

        self.action_space = gym.spaces.MultiDiscrete([6, 2, ]) # drive controls (straight, left, right, drift left, drift right), item

    def _act(self, action):

        if type(action) == int and action == 0:
            return # this handles the no-ops done at the start of the atari wrapper, otherwise, everything should be an array

        # it's just easier to clear all of the keys first and then set the ones we want instead of trying to figure out which ones are already set
        self.emu.input.keypad_rm_key(Keys.NO_KEY_SET)

        if action[IDX_DRIVE] > ACTION_NOOP:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_A))

        if action[IDX_DRIVE] == ACTION_ACCELERATE_LEFT:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))
        elif action[IDX_DRIVE] == ACTION_ACCELERATE_RIGHT:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

        if action[IDX_ITEM] == ACTION_ITEM:
            self.emu.input.keypad_add_key(keymask(Keys.KEY_X))


def create_wrapped_mario_kart_env(render_mode):
    env = MarioKartEnv(include_lower_frame=True)
    env = MarioAtariWrapper(env, screen_size=84, frame_skip=4)
    #env = FrameStack(env, num_stack=4)

    return env

# register the environment with gym so that we can use it with stable-baselines

gym.envs.registration.register(
    id='MarioKartDS-v0',
    entry_point='kart_env:create_wrapped_mario_kart_env',
    nondeterministic=True,
)


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
