import ast
import argparse
import logging

import cv2
import os
import numpy as np

from models.ddpg_vae.model import DDPG_V2
from utils.env import launch_env
from vae.utils import load_vae
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, VaeWrapper


def _enjoy():          
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-450.pkl', type=str)
    parser.add_argument('-m', '--model', help="Path for trained ddpg model", default='results/ddpg_vae_70.pkl', type=str)
    args = parser.parse_args()
    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")
    vae = load_vae(args.vae)
    env = VaeWrapper(env, vae)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = DDPG_V2.load(args.model)

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = model.predict(np.array(obs))[0]
            # Perform action
            print(action)
            obs, reward, done, _ = env.step(action)
            cv2.imshow('1', vae.decode(obs)[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()
        done = False
        obs = env.reset()        

if __name__ == '__main__':
    _enjoy()