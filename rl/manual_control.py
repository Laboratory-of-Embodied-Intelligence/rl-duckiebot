#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import time
import os
import cv2
from pyglet.window import key
import numpy as np
import gym
from vae.utils import load_vae
from gym_duckietown.envs import DuckietownEnv
from PIL import Image
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, VaeWrapper
# from experiments.utils import save_img


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-450.pkl', type=str)
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dataset_gen_path', help='Path for storing dataset of images. If empty, dataset is not recorded')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

class updater:
    def __init__(self, n_episodes):
        self.i = 0

        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_returns = np.zeros((n_episodes,))
        self.episode_starts  = []
        self.episode_reward = 0
        self.episodes_played = 0
        self.max_episodes = n_episodes

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.5, 0.5])
        if key_handler[key.DOWN]:
            action = np.array([-0.5, -0.5])
        if key_handler[key.LEFT]:
            action = np.array([0.15, +0.35])
        if key_handler[key.RIGHT]:
            action = np.array([0.15, -0.35])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = env.step(action)

        self.episode_reward += reward
        print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))



        self.actions.append(action)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.episode_starts.append(done)

        if args.dataset_gen_path:
            if not os.path.exists(args.dataset_gen_path):
                os.makedirs(args.dataset_gen_path)
            args.dataset_gen_path = args.dataset_gen_path if args.dataset_gen_path[-1]=='/'\
                                        else args.dataset_gen_path + '/'        
            im = Image.fromarray(obs)

            im.save(args.dataset_gen_path + f'obs_{self.i}.png')
            self.i+=1

        if done:
            print('done!')
            print(f"Episode reward {self.episode_reward}")
            self.episode_returns[self.episodes_played] = self.episode_reward
            self.episodes_played += 1
            self.episode_reward = 0
            env.reset()
            env.render()
        
        if self.episodes_played == self.max_episodes:
            print("Terminating because max_episodes reached")
            #time.sleep(4)
            self.rewards        = np.array(self.rewards)
            self.episode_starts = np.array(self.episode_starts[:-1])
            #self.actions        = np.concatenate(self.actions).reshape((-1,) + env.action_space.shape)
            #self.observations   = np.concatenate(self.observations).reshape((-1,) + env.observation_space.shape)
            self.observations   = np.array(self.observations)
            self.observations   = self.observations.reshape((self.observations.shape[0], self.observations.shape[-1]))
            self.actions        = np.array(self.actions)
            print(self.observations.shape)
            print(self.observations[0].shape)
            assert len(self.observations) == len(self.actions)

            numpy_dict = {
                'actions': self.actions,
                'obs':     self.observations,
                'rewards': self.rewards,
                'episode_returns': self.episode_returns,
                'episode_starts':  self.episode_starts
            }  # type: Dict[str, np.ndarray]
            #print(numpy_dict)
            np.savez("dataset.npz", **numpy_dict)
            env.close()
            sys.exit(0)

        env.render()

            

vae = load_vae(args.vae)
env = VaeWrapper(env, vae)
uw = updater(50)
pyglet.clock.schedule_interval(uw.update, 1.0 / env.unwrapped.frame_rate)


# Enter main event loop
pyglet.app.run()

env.close()
