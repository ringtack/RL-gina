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

tf.config.list_physical_devices("GPU")


def openai_atari_model(env_id):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(env)


def evaluate(model, env1, env2, vid1_dir, vid2_dir, vid1_name, vid2_name, step, vid=True):
    state = env1.reset()
    env1.render()
    done = False

    if vid:
        frames1 = []

    reward1 = 0.0
    t1 = 0
    while not done:
        if vid:
            frames1.append(Image.fromarray(env1.render(mode="rgb_array")))
        action = model.act(state, t1)
        state, rwd, done, _ = env1.step(action)
        reward1 += rwd
        t1 += 1
        env1.render()
    print(f"Env1 Episode reward: {reward1}\t Episode length: {t1}")

    if vid:
        filename = f"{vid1_dir}/{vid1_name}-{step}.gif"
        with open(filename, "wb+") as f:
            im = frames1[0]
            im.save(
                f, save_all=True, optimize=False, duration=40, append_images=frames1[1:]
            )
            print(f"Saved game gif at step {step} to {filename}.")

    state = env2.reset()
    env2.render()
    done = False

    if vid:
        frames2 = []
    
    reward2 = 0.0
    t2 = 0

    while not done:
        if vid:
            frames2.append(Image.fromarray(env2.render(mode="rgb_array")))
        action = model.act(state, t2)
        state, rwd, done, _ = env2.step(action)
        reward2 += rwd
        t2 += 1
        env2.render()
    print(f"Env2 Episode reward: {reward2}\t Episode length: {t2}")

    if vid:
        filename = f"{vid2_dir}/{vid2_name}-{step}.gif"
        with open(filename, "wb+") as f:
            im = frames2[0]
            im.save(
                f, save_all=True, optimize=False, duration=40, append_images=frames2[1:]
            )
            print(f"Saved game gif at step {step} to {filename}.")

    return reward1, reward2, t1, t2


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


def load_model(filename, env1, env2, exp, stack):
    """
    Loads weights and experiences from specified file. Do not include extensions!
    """
    model = Agent(env1, env2, stack)

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
        model.initialize_experiences(env1, env2)

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


def train(model, env1, env2, eval_env1, eval_env2, steps, args, viz_states, vid=True):
    # Load args names
    weight_dir = args.weight_dir
    weight_name = args.weight_name
    
    rwd1_dir = args.rwd1_dir
    rwd1_name = args.rwd1_name
    viz1_dir = args.viz1_dir
    viz1_name = args.viz1_name
    td1_name = args.td1_loss
    vid1_dir = args.vid1_dir
    vid1_name = args.vid1_name
    
    rwd2_dir = args.rwd2_dir
    rwd2_name = args.rwd2_name
    viz2_dir = args.viz2_dir
    viz2_name = args.viz2_name
    td2_name = args.td2_loss
    vid2_dir = args.vid2_dir
    vid2_name = args.vid2_name

    # Initialize experience buffer if not already
    if steps == 1 and len(model.experience) < model.experience.capacity:
        model.initialize_experiences(env1, env2)


    # Store episode rewards and lengths for both games
    episode1_rewards = [0.0]
    episode1_lengths = [0]

    # Provide constant csv writer for evaluation
    eval1_writer = csv.writer(open(f"{rwd1_dir}/{rwd1_name}-evals.csv", "w+", newline=""))
    eval1_writer.writerow(["episode_rewards", "episode_lengths"])

    # Write average Q-values to csv file
    viz1_writer = csv.writer(open(f"{viz1_dir}/{viz1_name}.csv", "w+", newline=""))
    viz1_writer.writerow(["episode", "sum q_values", "average q_values"])

    # Write losses to csv file
    td1_writer = csv.writer(open(f"{viz1_dir}/{td1_name}.csv", "w+", newline=""))
    td1_writer.writerow(["step", "loss value"])

    # Store episode rewards and lengths for both games
    episode2_rewards = [0.0]
    episode2_lengths = [0]

    # Provide constant csv writer for evaluation
    eval2_writer = csv.writer(open(f"{rwd2_dir}/{rwd2_name}-evals.csv", "w+", newline=""))
    eval2_writer.writerow(["episode_rewards", "episode_lengths"])

    # Write average Q-values to csv file
    viz2_writer = csv.writer(open(f"{viz2_dir}/{viz2_name}.csv", "w+", newline=""))
    viz2_writer.writerow(["episode", "sum q_values", "average q_values"])

    # Write losses to csv file
    td2_writer = csv.writer(open(f"{viz2_dir}/{td2_name}.csv", "w+", newline=""))
    td2_writer.writerow(["step", "loss value"])

    # Store max reward for each game; if we want to store video, store frames as well
    max1_reward = 0.0
    max2_reward = 0.0
    if vid:
        frames1 = []
        frames2 = []

    # Start training! Train for NUM_EPISODES steps
    state1 = env1.reset()
    state2 = env2.reset()

    for t in range(steps, NUM_EPISODES + 1):
        # Handle keyboard interrupt
        try:
            # Every SAVE_FREQ steps, save model weights and rewards
            if t % SAVE_FREQ == 0:
                save_model(model, t, weight_dir, weight_name)
                save_rewards(episode1_rewards, episode1_lengths, rwd1_dir, rwd1_name)
                save_rewards(episode2_rewards, episode2_lengths, rwd2_dir, rwd2_name)

            # if we want video, store frame
            if vid:
                frames1.append(Image.fromarray(env1.render(mode="rgb_array")))
                frames2.append(Image.fromarray(env2.render(mode="rgb_array")))

            with tf.GradientTape() as tape:
                # Get next action
                action1 = model.act(state1, t)
                action2 = model.act(state2, t)

                # Get next state, and add to experience buffer
                next_state1, reward1, done1, _ = env1.step(action1)
                model.remember(state2, action1, reward1, next_state1, done1)
                state1 = next_state1

                next_state2, reward2, done2, _ = env2.step(action2)
                model.remember(state2, action2, reward2, next_state2, done2)
                state2 = next_state2

                # Update episode reward and length; if done, reset game
                episode1_rewards[-1] += reward1
                episode1_lengths[-1] += 1

                episode2_rewards[-1] += reward2
                episode2_lengths[-1] += 1

                if done1:
                    print(
                        f"Episode for 1st Game complete. Average reward: {episode1_rewards[-1] / episode1_lengths[-1]}"
                    )
                    print(
                        f"\tReward: {episode1_rewards[-1]}\tEpisode length: {episode1_lengths[-1]}"
                    )
                    state1 = env1.reset()
                    # If max reward is less than current episode reward, update
                    if max1_reward < episode1_rewards[-1]:
                        max1_reward = episode1_rewards[-1]
                        # If we want, save gif of max reward
                        if vid:
                            filename1 = f"{vid1_dir}/{vid1_name}_max.gif"
                            with open(filename1, "wb+") as f:
                                im = frames1[0]
                                im.save(
                                    f,
                                    save_all=True,
                                    #  optimize=True,
                                    duration=40,
                                    append_images=frames1[1:],
                                )
                                print(
                                    "=============================================================================="
                                )
                                print(
                                    f"==========New record! Saved to {filename1}.=========="
                                )
                                print(
                                    "=============================================================================="
                                )
                    if vid:
                        frames1 = []

                    sum_q, avg_q = get_qvals(model, viz_states)
                    viz1_writer.writerow([len(episode1_rewards), sum_q, avg_q])
                    episode1_rewards.append(0.0)
                    episode1_lengths.append(0)

                if done2:
                    print(
                        f"Episode for 2nd Game complete. Average reward: {episode2_rewards[-1] / episode2_lengths[-1]}"
                    )
                    print(
                        f"\tReward: {episode2_rewards[-1]}\tEpisode length: {episode2_lengths[-1]}"
                    )
                    state2 = env2.reset()
                    # If max reward is less than current episode reward, update
                    if max2_reward < episode2_rewards[-1]:
                        max2_reward = episode2_rewards[-1]
                        # If we want, save gif of max reward
                        if vid:
                            filename2 = f"{vid2_dir}/{vid2_name}_max.gif"
                            with open(filename2, "wb+") as f:
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
                                    f"==========New record! Saved to {filename2}.=========="
                                )
                                print(
                                    "=============================================================================="
                                )
                    if vid:
                        frames2 = []

                    sum_q, avg_q = get_qvals(model, viz_states)
                    viz2_writer.writerow([len(episode2_rewards), sum_q, avg_q])
                    episode2_rewards.append(0.0)
                    episode2_lengths.append(0)


                if t % LEARNING_FREQ == 0:
                    loss = model.optimize_model()
                    if t % (LEARNING_FREQ * 10) == 0:
                        td1_writer.writerow([t, loss.numpy()])
                        td2_writer.writerow([t, loss.numpy()])
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
                reward1, reward2, length1, length2= evaluate(model, eval_env1, eval_env2, vid1_dir, vid2_dir, vid1_name, vid2_name, t)
                eval1_writer.writerow([reward1, length1])
                eval2_writer.writerow([reward2, length2])
        except KeyboardInterrupt:
            save_model(model, t, weight_dir, weight_name)
            save_rewards(episode1_rewards, episode1_lengths, rwd1_dir, rwd1_name)
            save_rewards(episode2_rewards, episode2_lengths, rwd2_dir, rwd2_name)
            sys.exit(1)


def load_states(state_file):
    with open(state_file, "rb") as p_f:
        return pickle.load(p_f)


def main(args):
    print(args)
    env1_id = f"{args.env1}NoFrameskip-v4"
    env1 = make_atari_model(env1_id, clip_rewards=False, frame_stack=args.stack)
    eval_env1 = make_atari_model(
        env1_id, clip_rewards=False, episode_life=False, frame_stack=args.stack
    )

    env2_id = f"{args.env2}NoFrameskip-v4"
    env2 = make_atari_model(env2_id, clip_rewards=False, frame_stack=args.stack)
    eval_env2 = make_atari_model(
        env2_id, clip_rewards=False, episode_life=False, frame_stack=args.stack
    )

    if args.load:
        model, steps = load_model(args.load, env1, env2, args.exp, args.stack)
    else:
        model = Agent(env1, env2, args.stack)
        steps = 1
    if args.steps != 1:
        steps = args.steps

    viz_states = load_states(args.states)

    print("Starting training...")
    train(model, env1, env2, eval_env1, eval_env2, steps, args, viz_states)
    print(f"Training done. Evaluating after {NUM_EPISODES} steps...")

    print("Type anything to evaluate model (EOF to exit)")
    for _ in sys.stdin:
        reward1, reward2, length1, length2 = evaluate(model, eval_env1, eval_env2, args.vid1_dir, args.vid2_dir, args.vid1_name, args.vid2_name, "END")
        print(f"Reward: {reward1}\tEpisode Length: {length1}")
        print(f"Reward: {reward2}\tEpisode Length: {length2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--env1",
        action="store",
        #  default="SpaceInvaders",
        #  default="VideoPinball",
        default="DemonAttack",
        help="Atari game (supported games: Pong, Cartpole, SpaceInvaders, Breakout, BeamRider)",
    )

    parser.add_argument(
        "--env2",
        action="store",
        #  default="SpaceInvaders",
        #  default="VideoPinball",
        default="SpaceInvaders",
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
        "--vid1_dir",
        action="store",
        default="./gifs",
        help="Set video directory",
    )

    parser.add_argument(
        "--vid1_name",
        action="store",
        default=time.strftime("vid1-%m-%d-%H-%M"),
        help="Set video name",
    )

    parser.add_argument(
        "--vid2_dir",
        action="store",
        default="./gifs",
        help="Set video directory",
    )

    parser.add_argument(
        "--vid2_name",
        action="store",
        default=time.strftime("vid2-%m-%d-%H-%M"),
        help="Set video name",
    )

    parser.add_argument(
        "--rwd1_dir", action="store", default="./rwds", help="Set reward directory"
    )

    parser.add_argument(
        "--rwd1_name",
        action="store",
        default=time.strftime("reward1-%m-%d-%H-%M"),
        help="Set reward file name",
    )

    parser.add_argument(
        "--rwd2_dir", action="store", default="./rwds", help="Set reward directory"
    )

    parser.add_argument(
        "--rwd2_name",
        action="store",
        default=time.strftime("reward2-%m-%d-%H-%M"),
        help="Set reward file name",
    )

    parser.add_argument("--load", action="store", default="", help="Load model")
    parser.add_argument(
        "--exp", action="store_true", default=False, help="Load experience"
    )
    parser.add_argument(
        "--stack", action="store_true", default=False, help="Frame stack"
    )

    parser.add_argument(
        "--states",
        action="store",
        default="viz/q_states.pickle",
        help="Visualization frames",
    )

    parser.add_argument(
        "--viz1_dir", action="store", default="viz", help="Visualization directory"
    )

    parser.add_argument(
        "--viz1_name",
        action="store",
        default=time.strftime("qvals-%m-%d-%H-%M"),
        help="Average Q-Value information",
    )

    parser.add_argument(
        "--viz2_dir", action="store", default="viz", help="Visualization directory"
    )

    parser.add_argument(
        "--viz2_name",
        action="store",
        default=time.strftime("qvals-%m-%d-%H-%M"),
        help="Average Q-Value information",
    )

    parser.add_argument(
        "--td1_loss",
        action="store",
        default=time.strftime("td1-loss-%m-%d-%H-%M"),
        help="TD Loss(?) information",
    )

    parser.add_argument(
        "--td2_loss",
        action="store",
        default=time.strftime("td2-loss-%m-%d-%H-%M"),
        help="TD Loss(?) information",
    )

    parser.add_argument(
        "--steps",
        action="store",
        default=1,
        help="Specify number of steps [TESTING ONLY]",
    )
    main(parser.parse_args())
