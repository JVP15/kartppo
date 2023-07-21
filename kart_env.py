from datetime import datetime

import gymnasium as gym
import numpy as np
import cv2


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import RecurrentPPO

from desmume.emulator import DeSmuME, SCREEN_PIXEL_SIZE, SCREEN_PIXEL_SIZE_BOTH, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_HEIGHT_BOTH
from desmume.controls import Keys, keymask
import torch
from torch import nn

TOTAL_LAPS = 3


# addresses from https://tasvideos.org/GameResources/DS/MarioKartDS
PLAYER_DATA_ADDR = 0x217ACF8
PLAYER_SPEED_OFFSET = 0x2A8

CHECKPOINT_DATA_ADDR = 0x021661B0
CHECKPOINT_OFFSET = 0xDAE
PLACE_OFFSET_ADDR = 0x3D8

ACTION_NOOP = 0
ACTION_ACCELERATE = 1
ACTION_ACCELERATE_ITEM = 2
ACTION_ACCELERATE_LEFT = 3
ACTION_ACCELERATE_RIGHT = 4

FRAMES_BEFORE_START = 256 # there is the 3, 2, 1 countdown, although 256 isn't the exact value, it's close enough.
TIMEOUT = 600 # if we don't pass a checkpoint after 600 frames (~10 seconds @ Mario Kart's naitive 60fps) then we terminate the episode

ROM_FILE = 'Mario Kart DS (USA) (En,Fr,De,Es,It)/Mario Kart DS (USA) (En,Fr,De,Es,It).nds'
#SAVESTATE_FILES = ('figure_8_time_trial.dsv', )
SAVESTATE_FILES = ('figure_8_time_trial.dsv', 'yoshi_falls_time_trial.dsv', 'cheep_cheep_beach_time_trial.dsv', 'luigis_mansion_time_trial.dsv')
#SAVESTATE_FILES = ('luigis_mansion_time_trial.dsv', )
#SAVESTATE_FILES = ('rainbow_road_time_trial.dsv', )


class MarioKartEnv(gym.Env):

    def __init__(self, include_lower_frame=False, rom_file=ROM_FILE, savestate_files=SAVESTATE_FILES):
        super().__init__()

        self.include_lower_frame = include_lower_frame

        self.action_space = gym.spaces.Discrete(5)  # no op, accelerate, accelerate + left, accelerate + right

        if self.include_lower_frame:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 3),
                                                    dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                                                    dtype=np.uint8)

        self.emu = DeSmuME()
        self.emu.volume_set(0)  # emulator makes noise even when headless

        self.mem = self.emu.memory
        try:
            self.emu.open(rom_file)
        except:
            print('Could not open ROM file ({}).'.format(rom_file))
            exit(1)
        self.savestate_files = savestate_files
        self.track = 0

        self.checkpoint = None
        self.last_checkpoint = None  # this is useful for detecting when we've finished a lap; we set this during reset()
        self.frames_since_checkpoint = -FRAMES_BEFORE_START  # if we don't make any forward progress, we're stuck and we terminate the episode (also give a buffer for the countdown)

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
            screen = screen_pixels[
                     :SCREEN_PIXEL_SIZE_BOTH * 4]  # see https://py-desmume.readthedocs.io/en/latest/quick_start.html#custom-drawing
            screen = screen.reshape((SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4))[..., :3]  # drop the alpha channel
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
        # choose a random number to select the savestate
        self.track_num = np.random.randint(0, len(self.savestate_files))
        savestate_file = self.savestate_files[self.track_num]
        try:
            self.emu.savestate.load_file(savestate_file)
        except:
            print("Error loading savestate file: " + savestate_file)
            exit(1)

    def step(self, action):
        self._act(action)

        self.emu.cycle()

        cur_checkpoint = self._get_checkpoint()

        reward = -1 / 60 if self.frames_since_checkpoint >= 0 else 0  # lose 1 point per second after the race starts (we start at -256 frames to deal with the countdown, so we can use it here too)
        terminated = False
        truncated = False

        if (cur_checkpoint - self.checkpoint == 1  # if we've moved to the next checkpoint...
                or (cur_checkpoint == 0 and self.checkpoint == self.last_checkpoint)):  # or if we've finished a lap, give a reward
            # completing a whole lap is worth 100 points, so we divide by the number of checkpoints to get the reward per checkpoint
            # checkpoints range from [0, last_checkpoint], so normally we'd add 1 to the denominator, but we give a reward for passing the finish line, so we don't need to.
            reward = 200 / (self.last_checkpoint)

            if cur_checkpoint == 0 and self.checkpoint == self.last_checkpoint:  # once we've finished a lap increase the lap counter and give a larger reward
                self.laps += 1
                reward = 100

            self.checkpoint = cur_checkpoint
            self.frames_since_checkpoint = 0
        else:
            self.frames_since_checkpoint += 1

        if self.laps == TOTAL_LAPS:
            terminated = True
        elif self.frames_since_checkpoint > TIMEOUT:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {'checkpoint': self.checkpoint, 'laps': self.laps,
                                                                'last_checkpoint': self.last_checkpoint}

    def render(self, mode='human'):
        if self.render_mode == 'human':
            frame = self._get_obs()
            # resize the frame to make it 2x bigger
            if self.include_lower_frame:
                frame = cv2.resize(frame, (SCREEN_WIDTH * 2, SCREEN_HEIGHT_BOTH * 2), interpolation=cv2.INTER_NEAREST)
            else:
                frame = cv2.resize(frame, (SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Mario Kart', frame)
            cv2.waitKey(1)
        else:
            return self._get_obs()

    def reset(self, seed=None, options=None):
        # NOTE: when we reset, we have to start at a time when the track is already loaded so we can get our current checkpoint (needed so that we can find the number of checkpoints)
        super().reset(seed=seed, options=options)

        self._load_savestate()
        self.checkpoint = self._get_checkpoint()
        self.last_checkpoint = self.checkpoint
        self.frames_since_checkpoint = -FRAMES_BEFORE_START
        self.laps = -1

        return self._get_obs(), {}


class MarioKartTrackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # obs should be a numpy array of shape 84x84x1
        # lets put the ohv of the track number in the upper left corner

        track_num = self.unwrapped.track_num
        obs[0, :len(SAVESTATE_FILES), 0] = 0  # 0 out everything first
        obs[0, track_num, 0] = 255

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs should be a numpy array of shape 84x84x1
        # lets put the ohv of the track number in the upper left corner
        track_num = self.unwrapped.track_num

        obs[0, :len(SAVESTATE_FILES), 0] = 0  # 0 out everything first
        obs[0, track_num, 0] = 255

        return obs, reward, terminated, truncated, info


class CNNPerTrackFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, ):
        super().__init__(observation_space, features_dim=features_dim)

        self.num_tracks = len(SAVESTATE_FILES)

        self.heads = nn.ModuleList([NatureCNN(observation_space, features_dim) for _ in range(self.num_tracks)])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        indices = []

        for i in range(self.num_tracks):
            indices.append(torch.where(observations[:, -1, 0, i] == 1)[
                               0])  # using the last frame on the stack b/c it is the only frame with the OHV at the very start of the game (before the stack gets filled)

        features = []
        for i in range(self.num_tracks):
            if len(indices[i]) > 0:
                features.append(self.heads[i](observations[indices[i]]))

        unordered_features = torch.cat(features, dim=0)
        ordered_indices = torch.cat(indices, dim=0).unsqueeze(1).expand(-1, self.features_dim)

        reordered_features = torch.scatter(unordered_features, 0, ordered_indices, unordered_features)

        return reordered_features


if __name__ == '__main__':
    def make_env_with_stats():
        env = MarioKartEnv(include_lower_frame=True)
        env = AtariWrapper(env, clip_reward=False, terminal_on_life_loss=False)
        env = MarioKartTrackWrapper(env)
        env = Monitor(env)

        return env


    env = make_vec_env(make_env_with_stats, n_envs=16,
                       vec_env_cls=SubprocVecEnv, )
    env = VecFrameStack(env, n_stack=4)

    # get the current datetime so we can use it to name our tensorboard log directory
    now = datetime.now()
    output_folder = f'./runs/{now.strftime("%Y-%m-%d_%H-%M-%S")}/'

    # model = RecurrentPPO('CnnLstmPolicy', env, verbose=1, tensorboard_log=output_folder, policy_kwargs={'enable_critic_lstm': False}, n_steps=2048)
    policy_kwargs = {'features_extractor_class': CNNPerTrackFeatureExtractor}
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=output_folder, n_steps=2048, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=1_000_000)

    # now that we have a trained model, we can save it and load it later
    model.save(output_folder + 'mario-kart-ppo')