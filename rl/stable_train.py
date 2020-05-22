
import numpy as np
import argparse
import os
import models.utils
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, VaeWrapper
from vae.utils import load_vae
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import HER
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from utils.dataset import ExpertDataset
from models.ddpg_vae.model import DDPG_V2
from models.sac_vae.model import SACWithVAE
from teleop.teleop_client import TeleopEnv

if __name__=="__main__":    
 # Initialize tensorboard

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
    parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-450.pkl', type=str)
    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=250000, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.25, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.1, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.25, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--l2-reg", default = 1e-2, type=float)
    parser.add_argument("--noise_clip", default=0.25, type=float)  # Range to clip target policy noise
    parser.add_argument("--replay_buffer_max_size", default=35000, type=int)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument('--tb-dir', type=str, default=None,)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                   sigma=args.expl_noise * np.ones(n_actions))
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.3,desired_action_stddev=0.5, adoption_coefficient=1.5)
    vae = load_vae(args.vae)
    env = VaeWrapper(env, vae)
    
    dataset = ExpertDataset(expert_path='dataset.npz',
                        traj_limitation=1, batch_size=128)

    # First try DDPG with no pretrain, than ddpg with pretrain, sac, sac with pretrain
    model = DDPG_V2(policy = 'CustomDDPGPolicy',
                    env = env,
                    tensorboard_log = "runs/DDPG",
                    gamma = args.discount,
                    buffer_size = args.replay_buffer_max_size,
                    tau = args.tau,
                    action_noise = action_noise,
                    eval_env = env,
                    param_noise = param_noise,
                    critic_l2_reg = args.l2_reg, 
                    verbose=2,
                    random_exploration=0.1,
                    nb_rollout_steps=1000,
                    full_tensorboard_log = True,
                    normalize_observations=True,
                    normalize_returns=True)

    model.learn(args.max_timesteps, tb_log_name="runs/DDPG")
    model.save('runs/DDPG')
    del model 


    model = DDPG_V2(policy = 'CustomDDPGPolicy',
                    env = env,
                    tensorboard_log = "runs/DDPG_pretrain",
                    gamma = args.discount,
                    buffer_size = args.replay_buffer_max_size,
                    tau = args.tau,
                    action_noise = action_noise,
                    eval_env = env,
                    param_noise = param_noise,
                    critic_l2_reg = args.l2_reg,
                    verbose=2,
                    random_exploration=0.1,
                    full_tensorboard_log = True,
                    nb_rollout_steps=0,
                    normalize_observations=True,
                    normalize_returns=True)

    model = model.pretrain(dataset, n_epochs=10000)
    model.learn(args.max_timesteps, tb_log_name="runs/DDPG_pretrain")
    model.save('runs/DDPG_pretrain')
    del model 


    model = SACWithVAE(policy = 'CustomSACPolicy',
                       env = env, 
                       tensorboard_log = "runs/SAC",
                       buffer_size     = args.replay_buffer_max_size,
                       gamma           = args.discount,
                       action_noise    = action_noise,
                       verbose=2,
                       random_exploration=0.1,
                       learning_starts=1000,
                       full_tensorboard_log = True)
                    
    model.learn(args.max_timesteps, tb_log_name="runs/SAC")
    model.save('runs/SAC')
    del model 


    model = SACWithVAE(policy = 'CustomSACPolicy',
                       env = env, 
                       tensorboard_log = "runs/SAC_pretrain",
                       buffer_size     = args.replay_buffer_max_size,
                       gamma           = args.discount,
                       action_noise    = action_noise,
                       verbose=2,
                       random_exploration=0.1,
                       full_tensorboard_log = True,
                       learning_starts=0)
    model = model.pretrain(dataset, n_epochs=10000)
    model.learn(args.max_timesteps, tb_log_name="runs/SAC_pretrain")
    model.save('runs/SAC_pretrain')
    del model 

