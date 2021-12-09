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
    LEARNING_FREQ,
    NUM_EPISODES,
    TARGET_UPDATE,
)

tf.config.list_physical_devices('GPU')
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


def evaluate(model, env1, env2):
    state = env1.reset()
    env1.render()
    done1 = False

    reward = 0.0
    t = 0
    while not done1:
        tf_state = tf.convert_to_tensor(state, dtype=np.float32)
        tf_state = tf.expand_dims(tf_state, 0)
        print(tf_state)
        action_qvals = model.q_net(tf_state)
        action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]
        state, rwd, done1, _ = env1.step(action)
        reward += rwd
        t += 1
        env1.render()
    print(f"Episode reward Env 1: {reward}\t Episode length: {t}")

    state = env2.reset()
    env2.render()
    done2 = False

    reward = 0.0
    t = 0
    while not done2:
        tf_state = tf.convert_to_tensor(state, dtype=np.float32)
        tf_state = tf.expand_dims(tf_state, 0)
        print(tf_state)
        action_qvals = model.q_net(tf_state)
        action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]
        state, rwd, done2, _ = env2.step(action)
        reward += rwd
        t += 1
        env2.render()
    print(f"Episode reward Env 2: {reward}\t Episode length: {t}")


def train(model, env1, env2):
    model.initialize_experiences(env1, env2)
    episode_rewards1 = [0.0]
    episode_lengths1 = [0]

    episode_rewards2 = [0.0]
    episode_lengths2 = [0]

    state1 = env1.reset()
    state2 = env2.reset()
    for t in range(NUM_EPISODES):
        # Epsilon-greedy: randomly explore epsilon% times
        eps = epsilon(t)
        sample = random.random()

        with tf.GradientTape() as tape:
            if sample < eps:
                # Exploration
                action1 = env1.action_space.sample()
                action2 = env2.action_space.sample()
            else:
                # Exploitation
                tf_state1 = tf.convert_to_tensor(state1, dtype=np.float32)
                tf_state1 = tf.expand_dims(tf_state1, 0)
                print(tf_state1)
                tf_state2 = tf.convert_to_tensor(state2, dtype=np.float32)
                tf_state2 = tf.expand_dims(tf_state2, 0)
                print(tf_state2)

                action_qvals1 = model.q_net(tf_state1)
                action1 = tf.cast(tf.math.argmax(action_qvals1, 1), tf.int32).numpy()[0]

                action_qvals2 = model.q_net(tf_state2)
                action2 = tf.cast(tf.math.argmax(action_qvals2, 1), tf.int32).numpy()[0]

            # Get next state, and add to experience buffer
            next_state1, reward1, done1, _ = env2.step(action1)
            model.remember(state1, action1, reward1, next_state1, done1)
            state1 = next_state1

            next_state2, reward2, done2, _ = env2.step(action2)
            model.remember(state2, action2, reward2, next_state2, done2)
            state2 = next_state2

            # Update episode reward and length; if done, reset game
            episode_rewards1[-1] += reward1
            episode_lengths1[-1] += 1
            if done1:
                print(
                    f"Episode complete. Average reward: {episode_rewards1[-1] / episode_lengths1[-1]}"
                )
                print(
                    f"\tReward: {episode_rewards1[-1]}\tEpisode length: {episode_lengths1[-1]}"
                )
                state = env1.reset()
                episode_rewards1.append(0.0)
                episode_lengths1.append(0)

            episode_rewards2[-1] += reward2
            episode_lengths2[-1] += 1
            if done2:
                print(
                    f"Episode complete. Average reward: {episode_rewards2[-1] / episode_lengths2[-1]}"
                )
                print(
                    f"\tReward: {episode_rewards2[-1]}\tEpisode length: {episode_lengths2[-1]}"
                )
                state = env2.reset()
                episode_rewards2.append(0.0)
                episode_lengths2.append(0)

            if t % LEARNING_FREQ == 0:
                loss = model.optimize_model()
                print(f"Loss at {t} steps:", loss)

        if t % LEARNING_FREQ == 0:
            # Only learn the Q network; the target network is set, and not learned
            gradients = tape.gradient(loss, model.q_net.trainable_variables)
            model.optimizer.apply_gradients(
                zip(gradients, model.q_net.trainable_variables)
            )

        if t % TARGET_UPDATE == 0:
            print(f"Updating target net after {t} steps...")
            model.update_target_net()

        if t % EVAL_STEPS == 0:
            print(f"Evaluating model after {t} steps...")
            evaluate(model, env1, env2)
            state1 = env1.reset()
            state2 = env2.reset()


def main(args):
    print(args)
    env1_id = f"{args.env1}NoFrameskip-v4"
    env1 = make_atari_model(env1_id)

    env2_id = f"{args.env2}NoFrameskip-v4"
    env2 = make_atari_model(env2_id)
    # Both agents have same action space
    model = Agent(env1.action_space.n)

    print("Starting training...")
    train(model, env1, env2)
    print(f"Training done. Evaluating after {NUM_EPISODES} steps...")
    evaluate(model, env1, env2)


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
        "--env1",
        action="store",
        default="SpaceInvaders",
        #  default="CartPole",
        help="Atari game (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )
    parser.add_argument(
        "--env2",
        action="store",
        default="DemonAttack",
        #  default="CartPole",
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
