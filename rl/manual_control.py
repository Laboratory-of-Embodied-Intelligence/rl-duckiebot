#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import os
from pyglet.window import key
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from PIL import Image
# from experiments.utils import save_img


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
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

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

class updater:
    def __init__(self):
        self.i = 0

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """

        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = env.step(action)
        print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    
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
            env.reset()
            env.render()

        env.render()

uw = updater()
pyglet.clock.schedule_interval(uw.update, 1.0 / env.unwrapped.frame_rate)


# Enter main event loop
pyglet.app.run()

env.close()
