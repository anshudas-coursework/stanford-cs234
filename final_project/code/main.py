# -*- coding: UTF-8 -*-

import torch
import numpy as np
import random
from ppo import PPO
from config import PulserConfig
from MultiStageRocket import Pulser

if __name__ == "__main__":
    # Create the environment
    sd = 1
    env = Pulser(seed=sd, render_mode=None)
    
    # Configuration
    cfg = PulserConfig(seed=sd)

    # Randomization
    torch.random.manual_seed(sd)
    np.random.seed(sd)
    random.seed(sd)

    # train model
    model = PPO(env, config=cfg, seed=sd)
    model.run()
