import argparse
import pickle
import random
import sys
import time

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers.atari_preprocessing import AtariPreprocessing

from atari_wrappers import make_atari_model
from dqn import Agent


def openai_atari_model(env_id):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(env)


def main(args):
    print(args)
    env_id = f"{args.env}NoFrameskip-v4"
    env = make_atari_model(env_id)
    model = Agent(env.action_space.n)

    model.initialize_experiences(env)
    model.optimize_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Load debug files",
    )
    parser.add_argument(
        "--env",
        action="store",
        default="SpaceInvaders",
        help="Atari game (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )
    parser.add_argument(
        "--model",
        action="store",
        default="dqn",
        help="RL model (supported models: dqn, transformer, ppo, trpo, a2c, sac)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="Run evaluation with log only",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        default=False,
        help="Save images during evaluation",
    )

    #  parser.add_argument(
    #  "--model", action="store", default=None, help="Saved model filename"
    #  )

    parser.add_argument(
        "--name",
        action="store",
        default=time.strftime("%m-%d-%H-%M"),
        help="Save file name",
    )
    parser.add_argument(
        "--play", action="store_true", default=False, help="play with WSAD + Space"
    )
    parser.add_argument("--seed", action="store", type=int, help="PRNG seed")
    parser.add_argument("--test", action="store_true", default=False, help="run tests")
    parser.add_argument(
        "--view", action="store_true", default=False, help="View evaluation in a window"
    )
    parser.add_argument(
        "--weights", action="store_true", default=False, help="Print model weights"
    )

    main(parser.parse_args())
