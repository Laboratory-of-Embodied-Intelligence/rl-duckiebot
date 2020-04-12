import ast
import argparse
import logging

import cv2
import os
import numpy as np

from pyglet.window import key
from skimage import img_as_ubyte
from models.ddpg_vae.model import DDPG_V2
from utils.env import launch_env
from vae.utils import load_vae
from vae.controller import VAEController

from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, VaeWrapper



    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)




def _enjoy():          
    # Launch the env with our helper function
    env = launch_env()
    # Register a keyboard handler
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
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    print("Initialized environment")
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-250.pkl', type=str)
    parser.add_argument('-m', '--model', help="Path for trained ddpg model", default='results/ddpg_vae_70.pkl', type=str)
    args = parser.parse_args()
    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    #env = ActionWrapper(env)
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
            #print(action)
            #action = [0.5,0]
            obs, reward, done, _ = env.step(action)

            cv2.imshow('2', env.last_pre_encoded_obs)
            cv2.imshow('1', env.last_encoded_obs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            env.render()
        done = False
        obs = env.reset()        

if __name__ == '__main__':
    _enjoy()