import gymnasium as gym
import numpy as np

class AntGlobalPositionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box)

        self.pos_dim = 2

        low = np.append(env.observation_space.low, [-np.inf] * self.pos_dim)
        high = np.append(env.observation_space.high, [np.inf] * self.pos_dim)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        mj_data = self.env.unwrapped.data
        global_xy = mj_data.qpos[0:2].astype(observation.dtype)
        return np.concatenate([global_xy, observation])