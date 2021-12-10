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
from tensorflow.keras.models import model_from_json

from atari_wrappers import make_atari_model
from dqn import DDQN, Agent
from settings import (
    BATCH_SIZE,
    EPSILON_END,
    EPSILON_STEPS,
    EVAL_STEPS,
    LEARNING_FREQ,
    NUM_EPISODES,
    SAVE_FREQ,
    TARGET_UPDATE,
)
from shared_agent import SharedAgent

tf.config.list_physical_devices("GPU")


def openai_atari_model(env_id):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(env)


def evaluate(model, env, vid_dir, vid_name, step, vid=True, second=False):
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
        if second:
            _, action = model.act(state, state, t)
        else:
            action, _ = model.act(state, state, t)
        state, rwd, done, _ = env.step(action)
        reward += rwd
        t += 1
        env.render()
    print(f"Episode reward: {reward}\t Episode length: {t}")

    if vid:
        add_env2 = "_env2" if second else ""
        filename = f"{vid_dir}/{vid_name}-{step}{add_env2}.gif"
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
    filename = f"{weight_dir}/{weight_name}-{t}"

    # attempt with json
    #  with open(f"{filename}.json", "w+") as json_file:
    #  json_file.write(model.q_net.to_json())
    #  print(f"Saved model json to {filename}.json.")

    model.q_net.save_weights(f"{filename}.h5")
    print(f"Saved model weights to {filename}.h5.")
    pickled_filename = f"{filename}.pickle"
    with open(pickled_filename, "wb+") as f:
        pickle.dump([model.experience, t], f)
    print(f"Saved experience to {pickled_filename}.")
    return filename, pickled_filename


def load_model(filename, env, exp, stack, env2=None):
    """
    Loads weights and experiences from specified file. Do not include extensions!
    """
    if env2 is not None:
        model = SharedAgent(env, env2, stack)
    model = Agent(env, stack)

    print(f"Loading model from {filename}...")
    # Load model json
    #  with open(f"{filename}.json", "r") as json_file:
    #  loaded_model_json = json_file.read()
    #  model.q_net = model_from_json(loaded_model_json)
    #  print(f"Loaded model json from {filename}.json.")

    #  model.q_net.set_weights(tf.keras.models.load_weights(f"{filename}.h5"))

    if exp:
        with open(f"{filename}.pickle", "rb") as p_f:
            model.experience, steps = pickle.load(p_f)
        print(f"Loaded experience from {filename}.pickle.")
    else:
        print("Experience buffer not found. Initializing...")
        model.initialize_experiences(env)

    # Try to initialize the net
    model.optimize_model()
    #  model.q_net(np.zeros((1, 84, 84, 4)))

    # Load model weights
    model.q_net.load_weights(f"{filename}.h5")

    print(f"Loaded model weights from {filename}.h5:")
    model.q_net.summary()

    return model, steps if exp else 1


def save_rewards(episode_rewards, episode_lengths, rwd_dir, rwd_name):
    filename = f"{rwd_dir}/{rwd_name}.csv"
    with open(filename, "w+", newline="") as f:
        w = csv.writer(f)
        w.writerow(episode_rewards)
        w.writerow(episode_lengths)
        print(f"Saved current rewards to {filename}.")


def get_qvals(model, viz_states):
    viz_states_batch = tf.convert_to_tensor(viz_states, dtype=tf.float32)
    q_vals = model.q_net(viz_states_batch)
    sum_q = tf.reduce_sum(q_vals).numpy()
    avg_q = tf.reduce_mean(tf.reduce_sum(q_vals, axis=1)).numpy()
    print(
        f"\tSum Q: {tf.reduce_sum(q_vals).numpy()}\t\tAverage Q: {tf.reduce_mean(tf.reduce_sum(q_vals,axis=1)).numpy()}"
    )
    return sum_q, avg_q


def train(
    model,
    env,
    eval_env,
    steps,
    args,
    viz_states,
    vid=True,
    env2=None,
    eval_env2=None,
    viz_states2=None,
):
    # Load args names
    weight_dir = args.weight_dir
    weight_name = args.weight_name
    vid_dir = args.vid_dir
    vid_name = args.vid_name
    rwd_dir = args.rwd_dir
    rwd_name = args.rwd_name
    if env2:
        rwd_name2 = f"{args.rwd_name}-env2"
    viz_dir = args.viz_dir
    viz_name = args.viz_name
    td_name = args.td_loss

    # Initialize experience buffer if not already
    if steps == 1 and len(model.experience) < model.experience.capacity:
        model.initialize_experiences()
    # Store episode rewards and lengths
    episode_rewards = [0.0]
    episode_lengths = [0]
    if env2:
        episode_rewards2 = [0.0]
        episode_lengths2 = [0]

    # Provide constant csv writer for evaluation
    eval_writer = csv.writer(open(f"{rwd_dir}/{rwd_name}-evals.csv", "w+", newline=""))
    header = ["episode_rewards", "episode_lengths"]
    if env2:
        header += ["episode_rewards2", "episode_lengths2"]
    eval_writer.writerow(header)

    # Write average Q-values to csv file
    viz_writer = csv.writer(open(f"{viz_dir}/{viz_name}.csv", "w+", newline=""))
    header = ["episode1", "sum q_values", "average q_values"]
    viz_writer.writerow(header)
    if env2:
        viz_writer2 = csv.writer(
            open(f"{viz_dir}/{viz_name}-env2.csv", "w+", newline="")
        )
        header = ["episode2", "sum q_values2", "average q_values2"]
        viz_writer2.writerow(header)

    # Write losses to csv file
    td_writer = csv.writer(open(f"{viz_dir}/{td_name}.csv", "w+", newline=""))
    td_writer.writerow(["step", "loss value"])

    # Store max reward; if we want to store video, store frames as well
    max_reward = 0.0
    frames = []
    if env2:
        max_reward2 = 0.0
        frames2 = []

    # Start training! Train for NUM_EPISODES steps
    state = env.reset()
    if env2:
        state2 = env2.reset()

    for t in range(steps, NUM_EPISODES + 1):
        # Handle keyboard interrupt
        try:
            # Every SAVE_FREQ steps, save model weights and rewards
            if t % SAVE_FREQ == 0:
                save_model(model, t, weight_dir, weight_name)
                save_rewards(episode_rewards, episode_lengths, rwd_dir, rwd_name)
                save_rewards(episode_rewards2, episode_lengths2, rwd_dir, rwd_name2)

            # store frame
            frames.append(Image.fromarray(env.render(mode="rgb_array")))
            if env2:
                frames2.append(Image.fromarray(env2.render(mode="rgb_array")))

            with tf.GradientTape() as tape:
                # Get next action
                if env2:
                    action, action2 = model.act(state, state2, t)
                else:
                    action = model.act(state, t)

                # Get next state, and add to experience buffer
                next_state, reward, done, _ = env.step(action)
                model.remember(state, action, reward, next_state, done)
                if env2:
                    next_state2, reward2, done2, _ = env2.step(action2)
                    model.remember(state2, action2, reward2, next_state2, done2)

                # Update states
                state = next_state
                if env2:
                    state2 = next_state2

                # Update episode reward and length; if done, reset game
                episode_rewards[-1] += reward
                episode_lengths[-1] += 1
                if env2:
                    episode_lengths2[-1] += reward2
                    episode_lengths2[-1] += 1

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
                            filename = f"{vid_dir}/{vid_name}_max.gif"
                            with open(filename, "wb+") as f:
                                im = frames[0]
                                im.save(
                                    f,
                                    save_all=True,
                                    #  optimize=True,
                                    duration=40,
                                    append_images=frames[1:],
                                )
                                print(
                                    "=============================================================================="
                                )
                                print(
                                    f"==========New record! Saved to {filename}.=========="
                                )
                                print(
                                    "=============================================================================="
                                )
                    if vid:
                        frames = []

                    sum_q, avg_q = get_qvals(model, viz_states)
                    viz_writer.writerow([len(episode_rewards), sum_q, avg_q])
                    episode_rewards.append(0.0)
                    episode_lengths.append(0)

                if env2:
                    if done2:
                        print(
                            f"Episode 2 complete. Average reward: {episode_rewards2[-1] / episode_lengths2[-1]}"
                        )
                        print(
                            f"\tReward: {episode_rewards2[-1]}\tEpisode length: {episode_lengths2[-1]}"
                        )
                        state2 = env2.reset()
                        # If max reward is less than current episode reward, update
                        if max_reward2 < episode_rewards2[-1]:
                            max_reward2 = episode_rewards2[-1]
                            # If we want, save gif of max reward
                            if vid:
                                filename = f"{vid_dir}/{vid_name}-env2_max.gif"
                                with open(filename, "wb+") as f:
                                    im = frames2[0]
                                    im.save(
                                        f,
                                        save_all=True,
                                        #  optimize=True,
                                        duration=40,
                                        append_images=frames2[1:],
                                    )
                                    print(
                                        "=============================================================================="
                                    )
                                    print(
                                        f"==========New record [env 2]! Saved to {filename}.=========="
                                    )
                                    print(
                                        "=============================================================================="
                                    )
                        if vid:
                            frames2 = []

                        sum_q2, avg_q2 = get_qvals(model, viz_states2)
                        info = [len(episode_rewards2), sum_q2, avg_q2]

                        viz_writer2.writerow(info)
                        episode_rewards2.append(0.0)
                        episode_lengths2.append(0)

                if t % LEARNING_FREQ == 0:
                    loss = model.optimize_model()
                    if t % (LEARNING_FREQ * 10) == 0:
                        td_writer.writerow([t, loss.numpy()])
                    if t % (LEARNING_FREQ * 50) == 0:
                        print(f"Loss at step {t}: {loss.numpy()}")

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
                if env2:
                    reward2, length2 = evaluate(
                        model, eval_env2, vid_dir, f"{vid_name}-env2", t
                    )
                evals = [reward, length]
                if env2:
                    evals.append([reward2, length2])
                eval_writer.writerow(evals)
        except KeyboardInterrupt:
            save_model(model, t, weight_dir, weight_name)
            save_rewards(episode_rewards, episode_lengths, rwd_dir, rwd_name)
            if env2:
                save_rewards(episode_rewards2, episode_lengths2, rwd_dir, rwd_name2)
            sys.exit(1)


def load_states(state_file):
    with open(state_file, "rb") as p_f:
        return pickle.load(p_f)


def main(args):
    print(args)
    env_id = f"{args.env}NoFrameskip-v4"
    if args.env2:
        env2_id = f"{args.env2}NoFrameskip-v4"

    env = make_atari_model(
        env_id, clip_rewards=False, frame_stack=args.stack, max_episode_steps=1500
    )
    env2 = None
    if args.env2:
        env2 = make_atari_model(
            env2_id, clip_rewards=False, frame_stack=args.stack, max_episode_steps=1500
        )

    eval_env = make_atari_model(
        env_id,
        clip_rewards=False,
        episode_life=False,
        frame_stack=args.stack,
        max_episode_steps=5000,
    )
    eval_env2 = None
    if env2:
        eval_env2 = make_atari_model(
            env2_id,
            clip_rewards=False,
            episode_life=False,
            frame_stack=args.stack,
            max_episode_steps=5000,
        )

    if args.load:
        model, steps = load_model(args.load, env, args.exp, args.stack, env2=env2)
    else:
        if env2:
            model = SharedAgent(env, env2, args.stack)
        else:
            model = Agent(env, args.stack)
        steps = 1
    if args.steps != 1:
        steps = args.steps

    viz_states = load_states(args.states)
    viz_states2 = None
    if env2:
        viz_states2 = load_states(args.states2)

    print("Starting training...")
    train(
        model,
        env,
        eval_env,
        steps,
        args,
        viz_states,
        env2=env2,
        eval_env2=eval_env2,
        viz_states2=viz_states2,
    )

    print(f"Training done. Evaluating after {NUM_EPISODES} steps...")

    print("Type anything to evaluate model (EOF to exit)")
    for _ in sys.stdin:
        reward, length = evaluate(model, eval_env, args.vid_dir, args.vid_name, "END")
        print(f"Reward: {reward}\tEpisode Length: {length}")
        if env2:
            reward2, length2 = evaluate(
                model, eval_env2, args.vid_dir, f"{args.vid_name}-env2", "END"
            )
            print(f"Reward [Env 2]: {reward2}\tEpisode Length: {length2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--env",
        action="store",
        default="SpaceInvaders",
        #  default="VideoPinball",
        #  default="DemonAttack",
        help="Atari game (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )

    parser.add_argument(
        "--env2",
        action="store",
        default="",
        help="Specify second environment, if we want shared experience buffer.",
    )

    parser.add_argument(
        "--stack", action="store_true", default=False, help="Frame stack"
    )

    parser.add_argument(
        "--model",
        action="store",
        default="dddqn",
        help="RL model (supported models: dddqn)",
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
    parser.add_argument(
        "--exp", action="store_true", default=False, help="Load experience"
    )

    parser.add_argument(
        "--states",
        action="store",
        default="viz/q_states.pickle",
        help="Visualization frames",
    )
    parser.add_argument(
        "--states2",
        action="store",
        default="",
        help="Visualization frames for second environment",
    )

    parser.add_argument(
        "--viz_dir", action="store", default="viz", help="Visualization directory"
    )

    parser.add_argument(
        "--viz_name",
        action="store",
        default=time.strftime("qvals-%m-%d-%H-%M"),
        help="Average Q-Value information",
    )

    parser.add_argument(
        "--td_loss",
        action="store",
        default=time.strftime("td-loss-%m-%d-%H-%M"),
        help="TD Loss(?) information",
    )

    parser.add_argument(
        "--steps",
        action="store",
        default=1,
        help="Specify number of steps [TESTING ONLY]",
    )

    main(parser.parse_args())
