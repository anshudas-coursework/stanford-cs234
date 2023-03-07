import numpy as np
from envs.MultiStageRocket import Bounds, Dynamics

class PulserConfig:
    def __init__(self, seed=1):
        self.env_name = "Pulser-v1"
        self.record = False

        # Dynamcis and bounds
        dynamics = Dynamics
        self.sampling_bound_params = Bounds( [100.0, 1000.0], [0.0, np.pi], [-10.0, 10.0], [0.0, 100.0],
                                            [0.0, dynamics.mf], [0.0, 20.0], [0.0, 500], [0.0, 2*np.pi] )

        # Randomization
        self.seed = seed
        seed_str = "seed=" + str(seed)

        # output config
        self.output_path = "results/{}-{}/".format( self.env_name, seed_str )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.horizon = 100 # number of seconds in the episode
        self.num_batches = 300  # number of batches trained on
        self.batch_size = self.horizon / dynamics.update  # number of steps used to compute each policy update
        self.max_ep_len = int(self.sampling_bound_params.rt[1] / dynamics.update)  # maximum episode length
        self.learning_rate = 1e-4
        self.gamma = 1.00  # the discount factor
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 5
        self.layer_size = 256
        self.type = None # choose between 'None' or 'gru'

        # hyperparameters for PPO
        self.eps_clip = 0.2
        self.update_freq = 5
        self.eps = 1e-6

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size
