import argparse
import csv
import pickle
import random
import sys
import time

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from PIL import Image

from atari_wrappers import make_atari_model
from dqn import DDQN, Agent
from settings import (
    BATCH_SIZE,
    EPSILON_END,
    EVAL_STEPS,
    LEARNING_FREQ,
    NUM_EPISODES,
    SAVE_FREQ,
    TARGET_UPDATE,
)

tf.config.list_physical_devices("GPU")


def openai_atari_model(env_id):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(env)


def evaluate(model, env, vid_dir, vid_name, step, vid=True):
    state = env.reset()
    env.render()
    done = False

    if vid:
        frames = []

    reward = 0.0
    t = 0
    while not done:
        if vid:
            frames.append(Image.fromarray(env.render(mode="rgb_array")))
        action = model.act(state, t)
        state, rwd, done, _ = env.step(action)
        reward += rwd
        t += 1
        env.render()
    print(f"Episode reward: {reward}\t Episode length: {t}")

    if vid:
        filename = f"{vid_dir}/{vid_name}-{step}.gif"
        with open(filename, "wb+") as f:
            im = frames[0]
            im.save(
                f, save_all=True, optimize=False, duration=40, append_images=frames[1:]
            )
            print(f"Saved game gif at step {step} to {filename}.")

    return reward, t


def save_model(model, t, weight_dir, weight_name):
    """
    Saves weights and experiences to the specified file location.
    """
    filename = f"{weight_dir}/{weight_name}-{t}.h5"
    model.q_net.save_weights(filename)
    print(f"Saved model to {filename}.")
    pickled_filename = f"{weight_dir}/{weight_name}-{t}.pickle"
    with open(pickled_filename, "wb+") as f:
        pickle.dump([model.experience, t], f)
    print(f"Saved experience to {pickled_filename}.")
    return filename, pickled_filename


def load_model(filename, env):
    """
    Loads weights and experiences from specified file. Do not include extensions!
    """
    model = Agent(env, env.action_space.n)
    #  model.q_net = DDQN(env.action_space.n)
    # Prime the model for loading I guess lol
    model.q_net.build((None, 84, 84, 4))
    #  model.q_net(np.zeros((BATCH_SIZE, 84, 84, 4)))
    # Load model
    model.q_net.load_weights(f"{filename}.h5")
    #  model.q_net.set_weights(tf.keras.models.load_weights(f"{filename}.h5"))
    print(f"Loaded model weights from {filename}.h5:")
    model.summary()
    with open(f"{filename}.pickle", "rb") as p_f:
        model.experience, steps = pickle.load(p_f)
    print(f"Loaded experience from {filename}.pickle.")
    #  return model, steps
    return model, 1


def save_rewards(episode_rewards, episode_lengths, rwd_dir, rwd_name):
    filename = f"{rwd_dir}/{rwd_name}.csv"
    with open(filename, "w+", newline="") as f:
        w = csv.writer(f)
        w.writerow(episode_rewards)
        w.writerow(episode_lengths)
        print(f"Saved current rewards to {filename}.")


def train(model, env, eval_env, args, steps, vid=True):
    weight_dir = args.weight_dir
    weight_name = args.weight_name
    vid_dir = args.vid_dir
    vid_name = args.vid_name
    rwd_dir = args.rwd_dir
    rwd_name = args.rwd_name
    # Initialize experience buffer if not already
    if steps == 1:
        model.initialize_experiences(env)
    # Store episode rewards and lengths
    episode_rewards = [0.0]
    episode_lengths = [0]

    # Provide constant csv writer for evaluation
    eval_writer = csv.writer(f"{rwd_dir}/{rwd_name}-evals.csv", "w+", newline="")
    eval_writer.writerow(["episode_rewards", "episode_lengths"])

    # Store max reward; if we want to store video, store frames as well
    max_reward = 0.0
    if vid:
        frames = []

    # Start training! Train for NUM_EPISODES steps
    state = env.reset()
    for t in range(steps, NUM_EPISODES + 1):
        # Handle keyboard interrupt
        try:
            # Every SAVE_FREQ steps, save model weights and rewards
            if t % SAVE_FREQ == 0:
                save_model(model, t, weight_dir, weight_name)
                save_rewards(episode_rewards, episode_lengths, rwd_dir, rwd_name)

            # if we want video, store frame
            if vid:
                frames.append(Image.fromarray(env.render(mode="rgb_array")))

            with tf.GradientTape() as tape:
                # Get next action
                action = model.act(state, t)

                # Get next state, and add to experience buffer
                next_state, reward, done, _ = env.step(action)
                model.remember(state, action, reward, next_state, done)
                state = next_state

                # Update episode reward and length; if done, reset game
                episode_rewards[-1] += reward
                episode_lengths[-1] += 1

                if done:
                    print(
                        f"Episode complete. Average reward: {episode_rewards[-1] / episode_lengths[-1]}"
                    )
                    print(
                        f"\tReward: {episode_rewards[-1]}\tEpisode length: {episode_lengths[-1]}"
                    )
                    state = env.reset()
                    # If max reward is less than current episode reward, update
                    if max_reward < episode_rewards[-1]:
                        max_reward = episode_rewards[-1]
                        # If we want, save gif of max reward
                        if vid:
                            filename = f"{vid_dir}/max_reward.gif"
                            with open(filename, "wb+") as f:
                                im = frames[0]
                                im.save(
                                    f,
                                    save_all=True,
                                    #  optimize=True,
                                    duration=40,
                                    append_images=frames[1:],
                                )
                                print(f"New record! Saved to {filename}.")
                    if vid:
                        frames = []
                    episode_rewards.append(0.0)
                    episode_lengths.append(0)
                if t % LEARNING_FREQ == 0:
                    loss = model.optimize_model()
                    if t % (LEARNING_FREQ * 50) == 0:
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
                reward, length = evaluate(model, eval_env, vid_dir, vid_name, t)
                eval_writer.writerow([reward, length])
        except KeyboardInterrupt:
            save_model(model, t, weight_dir, weight_name)
            break


def main(args):
    print(args)
    env_id = f"{args.env}NoFrameskip-v4"
    env = make_atari_model(env_id, clip_rewards=False)
    eval_env = make_atari_model(env_id, clip_rewards=False, episode_life=False)

    if args.load:
        model, steps = load_model(args.load, env)
    else:
        model = Agent(env, env.action_space.n)
        steps = 1

    print("Starting training...")
    train(model, env, eval_env, steps, args)
    print(f"Training done. Evaluating after {NUM_EPISODES} steps...")

    reward, length = evaluate(model, env, args.vid_dir, args.vid_name, "END")
    print(f"Reward: {reward}\tEpisode Length: {length}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--env",
        action="store",
        #  default="SpaceInvaders",
        #  default="VideoPinball",
        default="DemonAttack",
        help="Atari game (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )

    parser.add_argument(
        "--model",
        action="store",
        default="dqn",
        help="RL model (supported models: dqn, ddqn, dddqn)",
    )

    parser.add_argument(
        "--weight_dir", action="store", default="./weights", help="Set weight directory"
    )

    parser.add_argument(
        "--weight_name",
        action="store",
        default=time.strftime("weight-%m-%d-%H-%M"),
        help="Save weight name",
    )

    parser.add_argument(
        "--vid_dir",
        action="store",
        default="./gifs",
        help="Set video directory",
    )

    parser.add_argument(
        "--vid_name",
        action="store",
        default=time.strftime("vid-%m-%d-%H-%M"),
        help="Set video name",
    )

    parser.add_argument(
        "--rwd_dir", action="store", default="./rwds", help="Set reward directory"
    )

    parser.add_argument(
        "--rwd_name",
        action="store",
        default=time.strftime("reward-%m-%d-%H-%M"),
        help="Set reward file name",
    )

    parser.add_argument("--load", action="store", default="", help="Load model")

    main(parser.parse_args())
