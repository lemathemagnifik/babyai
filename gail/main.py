#!/usr/bin/env python3

"""
Script to train agent through Generative Adversarial imitation learning using demonstrations.
"""


import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pickle
import blosc
import babyai
import math
import time
from copy import deepcopy
import os
import sys
import argparse

sys.path.append('/Users/leobix/babyai/gail/')

from gail.GAILmodel import imitationgail

# Parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser = ArgumentParser2()
parser.add_argument("--env", type=str, default="BabyAI-GoToRedBall-v0", help="name of the environment to train on")
parser.add_argument("--demos", type=str, default="/Users/leobix/demos/BabyAI-GoToRedBall-v0_agent.pkl", help="demos filename (REQUIRED or demos-origin or multi-demos required)")
parser.add_argument("--updates", type=int, default=10000,
                            help="maximum number of discriminator updates")
parser.add_argument("--lr-gen", type=float, default=1e-4,
                            help="learning rate of generator (default: 1e-4)")
parser.add_argument("--lr-disc", type=float, default=3e-4,
                            help="learning rate of discriminator (default: 3e-4)")
parser.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
parser.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
parser.add_argument("--eps", type=float, default=1e-8,
                            help="Adam optimizer epsilon (default: 1e-8)")
parser.add_argument("--gae-discount", type=float, default=0.95,
                            help="GAE discount factor (default: 0.95)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                            help="GAE lambda factor (default: 0.95)")
parser.add_argument("--batch-size", type=int, default=512,
                                help="batch size for PPO (default: 512)")
parser.add_argument("--mini-batch-size", type=int, default=64,
                                help="batch size for PPO (default: 64)")
parser.add_argument("--ppo-epochs", type=int, default=6,
                                help="number of PPO epochs (default: 6)")
parser.add_argument("--eval-set-size", type=int, default=100,
                                help="(evaluation set sizedefault: 100)")
parser.add_argument("--eval-num", type=int, default=10,
                                help="evaluation every eval-num updates (default: 10)")
parser.add_argument("--absorbing", type=bool, default=True,
                                help="add absorbing states after reaching the goal (default: True)")                                         


def main(args):
    # Verify the arguments when we train on multiple environments
    # No need to check for the length of len(args.multi_env) in case, for some reason, we need to validate on other envs
    #device
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    imitationgail(args)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
