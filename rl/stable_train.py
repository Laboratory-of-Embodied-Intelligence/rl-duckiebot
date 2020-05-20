
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
    #env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    

    tensorboard_log = 'runs'
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vae', help="Path for trained vae model", default='logs/vae-128-450.pkl', type=str)
    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=10000000, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.25, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.1, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.25, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--l2-reg", default = 1e-2, type=float)
    parser.add_argument("--noise_clip", default=0.25, type=float)  # Range to clip target policy noise
    parser.add_argument("--replay_buffer_max_size", default=75000, type=int)  # Maximum number of steps to keep in the replay buffer
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

   
    if (args.model_name):
        print(f"resuming {args.model_name}")
        model = DDPG_V2.load(args.model_name, env = env, reset_num_timesteps=False)
        
    else:
        # model = DDPG_V2(policy = 'CustomDDPGPolicy',env = env, tensorboard_log = tensorboard_log, verbose=0,
        #             gamma = args.discount,
        #             buffer_size = args.replay_buffer_max_size,
        #             tau = args.tau,
        #             action_noise = action_noise,
        #             eval_env = env,
        #             param_noise = param_noise,
        #             critic_l2_reg = args.l2_reg)
        model = SACWithVAE(policy = 'CustomSACPolicy',
                           env = env, 
                           tensorboard_log = tensorboard_log,
                           learning_rate   = 0.0003,
                           buffer_size     = 10000,
                           batch_size      = 64,
                           train_freq      = 301,
                           gamma           = 0.99,
                           ent_coef        = 'auto_0.1',
                           gradient_steps  = 600,
                           learning_starts = 300)
    #strategy = 'future'
    # Wrap around teleoperation mode
    #env = TeleopEnv(env, model,is_training=True)
    dataset = ExpertDataset(expert_path='dataset.npz',
                        traj_limitation=1, batch_size=128)
    print("Pretraining model")
    model.pretrain(dataset, n_epochs=10000)
    model.save('pretrained', cloudpickle=True)
    print("Starting training")
    #model = HER('MlpPolicy', env, model, n_sampled_goal=4, goal_selection_strategy=strategy, verbose=1)
    model.learn(args.max_timesteps, tb_log_name=args.tb_dir)
    #model.save(os.path.join("results/ddpg_vae"), cloudpickle=True)