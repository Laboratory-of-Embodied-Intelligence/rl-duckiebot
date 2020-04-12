# Implementation mostly taken from https://github.com/araffin/learning-to-drive-in-5-minutes/

import time
import os
import numpy as np
from mpi4py import MPI
from PIL import Image
from stable_baselines import logger
from stable_baselines.ddpg.ddpg import DDPG
from torch.utils.tensorboard import SummaryWriter
from models.utils import make_image

class DDPG_V2(DDPG):

    """
    Custom DDPG version in order to work with donkey car env.
    It is adapted from the stable-baselines version.
    Changes:
    - optimization is done after each episode
    - more verbosity.
    """
    def learn(self, total_timesteps, callback=None,
              log_interval=1, tb_log_name="DDPG", print_freq=100):
        with SummaryWriter(flush_secs=15) as writer:
            #writer.add_graph(self.graph)
            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)

            self.episode_reward = np.zeros((1,))
            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                actor_losses = []
                critic_losses = []
                should_return = False

                while True:
                    obs = self.env.reset()
                    # Rollout one episode.
                    while True:
                        if total_steps >= total_timesteps:
                            if should_return:
                                return self
                            should_return = True
                            break

                        # Predict next action.
                        action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                        if self.verbose >= 2:
                            print(action)
                        assert action.shape == self.env.action_space.shape

                        # Execute next action.
                        #if abs(action[0])>=1 or abs(action[1])>=1:
                        #    print("Wrong action")
                        #    sys.exit(-1)
                        
                        new_obs, reward, done, info = self.env.step(action * np.abs(self.action_space.low))

                        step += 1
                        total_steps += 1
                        if rank == 0 and self.render:
                            self.env.render()
                        episode_reward += reward
                        episode_step += 1

                        if print_freq > 0 and episode_step % print_freq == 0 and episode_step > 0:
                            print("{} steps".format(episode_step))

                        # Book-keeping.
                        self._store_transition(obs, action, reward, new_obs, done)

                        obs = new_obs
                        if callback is not None:
                            # Only stop training if return value is False, not when it is None. This is for backwards
                            # compatibility with callbacks that have no return statement.
                            if callback(locals(), globals()) is False:
                                return self

                        if done:
                            print("Episode finished. Reward: {:.2f} {} Steps".format(episode_reward, episode_step))

                            # Episode done.
                            log_reward = episode_reward
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            self._reset()
                            obs = self.env.reset()
                            # Finish rollout on episode finish.
                            break
                    # Train DDPG.
                    actor_losses = []
                    critic_losses = []
                    train_start = time.time()
                    for t_train in range(self.nb_train_steps):
                        critic_loss, actor_loss = self._train_step(0, None, log=t_train == 0)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        self._update_target_net()
                    print("DDPG training duration: {:.2f}s".format(time.time() - train_start))

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['train/loss_actor'] = np.mean(actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(critic_losses)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rewards/episode_reward'] = log_reward 

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}
                    
                    # Total statistics.
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        writer.add_scalar(key, combined_stats[key], step)
                        logger.record_tabular(key, combined_stats[key])

                    print(self.env.last_encoded_obs[::-1].shape)
                    image1 = Image.fromarray(np.uint8(self.env.last_encoded_obs))
                    image2 = Image.fromarray(np.uint8(self.env.last_pre_encoded_obs))
                    image1.save('debug1.jpg')
                    image2.save('debug2.jpg')
                    writer.add_image(str(step)+'_vae', self.env.last_encoded_obs, step, dataformats='HWC') 
                    writer.add_image(str(step)+'_true', self.env.last_pre_encoded_obs, step, dataformats='HWC') 

                    self.save(os.path.join(writer.log_dir+"/ddpg_vae_"  + str(episodes)), cloudpickle=True)

                    logger.dump_tabular()
                    logger.info('')


def as_scalar(scalar):
    """
    check and return the input if it is a scalar, otherwise raise ValueError
    :param scalar: (Any) the object to check
    :return: (Number) the scalar if x is a scalar
    """
    if isinstance(scalar, np.ndarray):
        assert scalar.size == 1
        return scalar[0]
    elif np.isscalar(scalar):
        return scalar
    else:
        raise ValueError('expected scalar, got %s' % scalar)