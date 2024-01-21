from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import cv2

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.atari_wrappers import StickyActionEnv, NoopResetEnv, MaxAndSkipEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.percent_complete_mean = 0
        self.place_mean = 9
        self.rollout_ended = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.logger.record('env/ep_%_done_mean', np.mean([info['percent_complete'] for info in self.model.ep_info_buffer]))
            self.logger.record('env/ep_place_mean', np.mean([info['place'] for info in self.model.ep_info_buffer]))
            self.rollout_ended = True
        return True

class MarioWarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, :]


class MarioAtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Based on Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * ~~Termination signal when a life is lost.~~ Removed b/c there are no lives
    * Resize to a square image: 84x84 by default
    * ~~Grayscale observation~~ Keeping RGB
    * ~~Clip reward to {-1, 0, 1}~~ Better to keep original reward
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        env = MarioWarpFrame(env, width=screen_size, height=screen_size)

        super().__init__(env)
