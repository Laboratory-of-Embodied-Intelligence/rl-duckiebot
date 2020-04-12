import gym
import numpy as np
from skimage import img_as_ubyte
from gym import spaces


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(160,80, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        #Image.fromarray(observation).show()\
        new_obs = np.array(Image.fromarray(observation).resize(self.shape[0:2]))
        return new_obs


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation#.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


class VaeWrapper(gym.ObservationWrapper):

    def __init__(self, env, vae):
        super(VaeWrapper, self).__init__(env)
        self.vae = vae

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                high=np.finfo(np.float32).max,
                                shape=(1, self.vae.z_size),
                                dtype=np.float32)
    def observation(self, observation):
        observation = img_as_ubyte(observation)
        self.last_pre_encoded_obs = observation
        enc_obs = self.vae.encode_from_raw_image(observation)
        self.last_encoded_obs = self.vae.decode(enc_obs)[0][:, :, ::-1]
        return enc_obs