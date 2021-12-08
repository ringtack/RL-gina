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
from settings import (
    EPSILON_END,
    EPSILON_START,
    EPSILON_STEPS,
    EVAL_STEPS,
    NUM_EPISODES,
    TARGET_UPDATE,
)


def openai_atari_model(env_id):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(env)


def epsilon(t):
    """
    Computes the epsilon value at a certain time step, following the linearly annealing
    epsilon-greedy strategy.
    """
    eps_frac = (EPSILON_END - EPSILON_START) / EPSILON_STEPS
    return max(EPSILON_END, EPSILON_START + t * eps_frac)


def evaluate(model, env):
    state = env.reset()
    env.render()
    done = False

    reward = 0.0
    t = 0
    while not done:
        tf_state = tf.convert_to_tensor(state, dtype=np.float32)
        tf_state = tf.expand_dims(tf_state, 0)
        action_qvals = model.q_net(tf_state)
        action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]
        state, rwd, done, _ = env.step(action)
        reward += rwd
        t += 1
        env.render()
    print(f"Episode reward: {reward}\t Episode length: {t}")


def train(model, env):
    model.initialize_experiences(env)
    episode_rewards = [0.0]
    episode_lengths = [0]

    state = env.reset()
    for t in range(NUM_EPISODES):
        # Epsilon-greedy: randomly explore epsilon% times
        eps = epsilon(t)
        sample = random.random()

        with tf.GradientTape() as tape:
            if sample < eps:
                # Exploration
                action = env.action_space.sample()
            else:
                tf_state = tf.convert_to_tensor(state, dtype=np.float32)
                tf_state = tf.expand_dims(tf_state, 0)
                action_qvals = model.q_net(tf_state)
                action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]

            next_state, reward, done, _ = env.step(action)
            model.remember(state, action, reward, next_state, done)
            state = next_state

            episode_rewards[-1] += reward
            episode_lengths[-1] += 1
            if done:
                state = env.reset()
                episode_rewards.append(0.0)
                episode_lengths.append(0)

            loss = model.optimize_model()

        # Only learn the Q network; the target network is set, and not learned
        gradients = tape.gradient(loss, model.q_net.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.q_net.trainable_variables))

        if len(episode_lengths) % 100 == 0:
            print(
                f"Current average episode reward: {sum(episode_rewards) / len(episode_rewards)}"
            )
        if t % TARGET_UPDATE == 0:
            print(f"Updating target net after {t} steps...")
            model.update_target_net()

        if t % EVAL_STEPS == 0:
            print(f"Evaluating model after {t} steps...")
            evaluate(model, env)


def main(args):
    print(args)
    env_id = f"{args.env}NoFrameskip-v4"
    env = make_atari_model(env_id)
    model = Agent(env.action_space.n)

    print("Starting training...")
    train(model, env)
    print(f"Training done. Evaluating after {NUM_EPISODES} steps...")
    evaluate(model, env)


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
