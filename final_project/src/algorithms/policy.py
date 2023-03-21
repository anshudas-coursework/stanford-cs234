import torch
import torch.nn as nn

from algorithms.network_utils import np2torch
from envs.RocketEnv import Pulser
from configs.PulserConfig import PulserConfig

class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)
        """
        temp_env = Pulser()
        observations = np2torch(observations)
        a = torch.clip(
            self.action_distribution(observations).sample(),
            torch.tensor([temp_env.sampling_bound_params.dV[0], temp_env.sampling_bound_params.phi[0]]),
            torch.tensor([temp_env.sampling_bound_params.dV[1], temp_env.sampling_bound_params.phi[1]])
        )
        sampled_actions = a.numpy()
        log_probs = self.action_distribution(observations).log_prob(a).detach().numpy()
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions

class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter( torch.zeros(action_dim) )

    def std(self):
        std = self.log_std.exp()
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()
        """
        env, config = Pulser(), PulserConfig()
        mean = 0.5*(self.network(observations)+1)*torch.tensor([env.sampling_bound_params.dV[1], env.sampling_bound_params.phi[1]])
        distribution = torch.distributions.MultivariateNormal(
            loc=mean+config.eps,
            scale_tril=(self.std()+config.eps).diag()
        )
        return distribution
