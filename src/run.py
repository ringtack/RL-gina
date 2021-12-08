import argparse
import pickle
import random
import sys
import time

import gym
import numpy as np
import tensorflow as tf

from atari_wrappers import make_atari, wrap_deepmind


def main(args):
    print(args)
    env = make_atari("{}NoFrameskip-v4".format(args.env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="load debug files and run fit_batch with them",
    )
    parser.add_argument(
        "--env",
        action="store",
        default="Breakout",
        help="Atari game name (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )
    parser.add_argument(
        "--model",
        action="store",
        default="dqn",
        help="model (supported models: dqn, transformer, ppo, trpo, a2c, sac)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="run evaluation with log only",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        default=False,
        help="save images during evaluation",
    )
    parser.add_argument(
        "--model", action="store", default=None, help="model filename to load"
    )
    parser.add_argument(
        "--name",
        action="store",
        default=time.strftime("%m-%d-%H-%M"),
        help="name for saved files",
    )
    parser.add_argument(
        "--play", action="store_true", default=False, help="play with WSAD + Space"
    )
    parser.add_argument(
        "--seed", action="store", type=int, help="pseudo random number generator seed"
    )
    parser.add_argument("--test", action="store_true", default=False, help="run tests")
    parser.add_argument(
        "--view", action="store_true", default=False, help="view evaluation in a window"
    )
    parser.add_argument(
        "--weights", action="store_true", default=False, help="print model weights"
    )

    main(parser.parse_args())
