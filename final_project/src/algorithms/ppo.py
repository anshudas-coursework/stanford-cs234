import numpy as np
import torch
from algorithms.general import export_plot
from algorithms.network_utils import np2torch
from algorithms.policy_gradient import PolicyGradient
import os

class PPO(PolicyGradient):

    def __init__(self, env, config, seed, logger=None):
        config.use_baseline = True
        super(PPO, self).__init__(env, config, seed, logger)
        self.eps_clip = self.config.eps_clip

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.

        Note:
            - PyTorch optimizers will try to minimize the loss you compute, but you
            want to maximize the policy's performance.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        self.optimizer.zero_grad()
        r = (self.policy.action_distribution(observations).log_prob(actions) - old_logprobs).exp()
        loss = -torch.mean( torch.minimum( r*advantages, torch.clamp(r, 1-self.config.eps_clip, 1+self.config.eps_clip)*advantages ) )
        loss.backward()
        self.optimizer.step()

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            # rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, 
                                   old_logprobs)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                    t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg) # type: ignore

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...") # type: ignore
                last_record = 0
                self.record()
            
            # Save model
            if np.remainder(t+1, 5)==0:
                export_plot(
                    averaged_total_rewards,
                    "Running Score",
                    self.config.env_name,
                    self.config.plot_output,
                )
            if np.remainder(t+1, 100)==0:
                if not os.path.exists(self.config.model_path):
                    os.makedirs(self.config.model_path)
                torch.save(self.policy.network, self.config.policy_output+"_"+str(t+1+self.config.last_iteration)+".weight")
                torch.save(self.policy.log_std, self.config.deviation_output+"_"+str(t+1+self.config.last_iteration)+".weight")
                np.save(self.config.scores_output, averaged_total_rewards)
            

        self.logger.info("- Training done.") # type: ignore

        # Save figure
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

        # Save model
        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)
        torch.save(self.policy.network, self.config.policy_output)
        torch.save(self.policy.log_std, self.config.deviation_output)


    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode = 0
        episode_rewards = [] 
        paths = []
        t = 0
        batch_idx = 0

        while num_episodes or batch_idx < self.config.batch_size:
            state = env.reset()
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob = True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    batch_idx += 1
                    break
                if (not num_episodes) and batch_idx == self.config.batch_size:
                    break
            
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards
