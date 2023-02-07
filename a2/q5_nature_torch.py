import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q3_schedule import LinearExploration, LinearSchedule
from q4_linear_torch import Linear
import logging


from configs.q5_nature import config


class NatureQN(Linear):
    """
    Implementing DQN that will solve MinAtar's environments.

    Model configuration can be found in the assignment PDF, section 4a.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. To calculate the size of the input to the first linear layer, you
               can use online tools that calculate the output size of a
               convolutional layer (e.g. https://madebyollin.github.io/convnet-calculator/)
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()

        ##############################################################
        ################ YOUR CODE HERE - 20-30 lines ################
        # Network constructor
        def create_sequential(self, n_out_channels, spatial_extent, n_hidden) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d( \
                    in_channels = n_channels * self.config.state_history, \
                    out_channels = n_out_channels, \
                    kernel_size = spatial_extent \
                ),
                nn.ReLU(), # 
                nn.Flatten(), # Use the default parameters for flatten (maintain batch dimension, flatten everything else)
                nn.Linear( \
                    in_features = (img_height-spatial_extent+1) * (img_width-spatial_extent+1) * n_out_channels, \
                    out_features = n_hidden \
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features = n_hidden, \
                    out_features = num_actions \
                )
            )
        self.q_network = create_sequential( self, 16, 3, 128 )
        self.target_network = create_sequential( self, 16, 3, 128 )
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        net = getattr( self, network )
        out = net( state.permute(0, 3, 1, 2) )
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == "__main__":
    logging.getLogger(
        "matplotlib.font_manager"
    ).disabled = True  # disable font manager warnings
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(
        env, config.eps_begin, config.eps_end, config.eps_nsteps
    )

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
