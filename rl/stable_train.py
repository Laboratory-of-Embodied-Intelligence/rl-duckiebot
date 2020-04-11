
import numpy as np
import argparse
import os
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, VaeWrapper
from vae.utils import load_vae
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from models.ddpg_vae.model import DDPG_V2
from models.utils import CustomDDPGPolicy

if __name__=="__main__":    
 # Initialize tensorboard

    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    

    tensorboard_log = 'runs'
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-100.pkl', type=str)
    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1000000, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.005, type=float)  # Range to clip target policy noise
    parser.add_argument("--replay_buffer_max_size", default=10000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--model-dir', type=str, default='reinforcement/pytorch/models/')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                   sigma=args.expl_noise * np.ones(n_actions))
    vae = load_vae(args.vae)
    env = VaeWrapper(env, vae)

    model = DDPG_V2(policy = 'CustomDDPGPolicy',env = env, tensorboard_log = tensorboard_log, verbose=0,
                    gamma = args.discount,
                    buffer_size = args.replay_buffer_max_size,
                    tau = args.tau,
                    action_noise = action_noise,
                    
                    render_eval = True)
    
    model.learn(args.max_timesteps)
    #model.save(os.path.join("results/ddpg_vae"), cloudpickle=True)