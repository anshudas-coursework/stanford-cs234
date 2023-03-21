from envs.RocketEnv import Dynamics

class PulserConfig:
    def __init__(self, env="Pulser", seed=1):
        self.env_name = env
        self.record = False

        # Dynamics and bounds
        dynamics = Dynamics

        # Randomization
        self.seed = seed
        seed_str = "seed=" + str(seed)

        # model and training config
        self.num_batches = 10000  # number of batches trained on
        self.max_ep_len = int(dynamics.max_time / dynamics.update)  # maximum episode length
        self.batch_size = 50  # number of episodes used to compute each policy update
        self.num_test_runs = 10
        self.learning_rate = 1e-4
        self.gamma = 1.00  # the discount factor
        self.normalize_advantage = True
        self.render_mode = 'none'
        self.running_mode = 'train'

        # output config
        self.output_path = "results/{}-{}/".format( self.env_name, seed_str )
        self.full_output_path = None
        self.model_path = self.output_path + "/model"
        self.policy_name = None
        self.last_iteration = 0
        self.policy_output = self.model_path + "/policy"
        self.deviation_output = self.model_path + "/stddev"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores_"+str(self.last_iteration)+"-"+str(self.last_iteration+self.num_batches)+".npy"
        self.plot_output = self.output_path + "scores_"+str(self.last_iteration)+"-"+str(self.last_iteration+self.num_batches)+".png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # parameters for the policy and baseline models
        self.n_layers = 7
        self.layer_size = 64
        self.type = None # choose between 'None' or 'gru'

        # hyperparameters for PPO
        self.eps_clip = 0.1
        self.update_freq = 10
        self.eps = 1e-9
