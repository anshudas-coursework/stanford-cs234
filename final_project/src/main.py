# -*- coding: UTF-8 -*-
import argparse
import torch
import numpy as np
import random
from algorithms.ppo import PPO
from configs.PulserConfig import PulserConfig
from envs.RocketEnv import Pulser

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--policy", type=int, default=0)
parser.add_argument("--run", type=str, default='test')
parser.add_argument("--opt_type", type=str, default='min_fuel')
parser.add_argument("--version", type=int, default=1)

parser.set_defaults(use_baseline=True)

## LOG ##
# - V2: increased opt rewards from 0.001 to 0.1
# - V3: changed dv_max from 50 to 200, opt reward from 0.1 to 0.5
# - V4:
# - V5: new NN (3 layers, 64 nodes each)

if __name__ == "__main__":
    args = parser.parse_args()

    # Configuration
    cfg = PulserConfig(seed=args.seed, env="PulserV"+str(args.version)+"_"+args.opt_type)
    if args.policy != 0:
        cfg.full_output_path = "C:/Users/Owner/Documents/coding/stanford-cs234/final_project/" + cfg.output_path
        cfg.policy_name = "_" + str(args.policy)
        cfg.last_iteration = args.policy
        cfg.scores_output = cfg.output_path + "scores_" + str(cfg.last_iteration) + "-" + str(cfg.last_iteration+cfg.num_batches) + ".npy"
        cfg.plot_output = cfg.output_path + "scores_" + str(cfg.last_iteration) + "-" + str(cfg.last_iteration+cfg.num_batches) + ".png"
    else:
        cfg.full_output_path = None
        cfg.policy_name = None
        cfg.last_iteration = 0
    
    if args.run == 'test':
        cfg.render_mode = 'human'
        cfg.running_mode = 'test'
    else:
        cfg.render_mode = 'none'
        cfg.running_mode = 'train'

    # Create the environment
    env = Pulser(seed=args.seed, render_mode=cfg.render_mode, opt_type=args.opt_type)

    # Randomization
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # train model
    model = PPO(env, config=cfg, seed=args.seed)
    if cfg.running_mode=='train': model.run()
    elif cfg.running_mode=='test': model.evaluate()
