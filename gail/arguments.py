"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default="BabyAI-GoToRedBall-v0",
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--demos", default="/Users/leobix/demos/BabyAI-GoToRedBall-v0_agent.pkl",
                    help="demos filename (REQUIRED or demos-origin or multi-demos required)")
        self.add_argument("--updates", type=int, default=10000,
                            help="maximum number of discriminator updates")
        self.add_argument("--lr-gen", type=float, default=1e-4,
                            help="learning rate of generator (default: 1e-4)")
        self.add_argument("--lr-disc", type=float, default=3e-4,
                            help="learning rate of discriminator (default: 3e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--optim-eps", type=float, default=1e-8,
                            help="Adam optimizer epsilon (default: 1e-8)")
        self.add_argument("--gae-discount", type=float, default=0.95,
                            help="GAE discount factor (default: 0.95)")
        self.add_argument("--gae-lambda", type=float, default=0.95,
                            help="GAE lambda factor (default: 0.95)")
        self.add_argument("--batch-size", type=int, default=512,
                                help="batch size for PPO (default: 512)")
        self.add_argument("--mini-batch-size", type=int, default=64,
                                help="batch size for PPO (default: 64)")
        self.add_argument("--ppo-epochs", type=int, default=6,
                                help="number of PPO epochs (default: 6)")
        self.add_argument("--eval-set-size", type=int, default=100,
                                help="(evaluation set sizedefault: 100)")
        self.add_argument("--eval-num", type=int, default=10,
                                help="evaluation every eval-num updates (default: 10)")
        self.add_argument("--absorbing", type=bool, default=True,
                                help="add absorbing states after reaching the goal (default: True)") 

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        return args
