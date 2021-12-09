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

tf.config.list_physical_devices('GPU')

def openai_atari_model(
    env_id,
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    terminal_on_life_loss=False,
    grayscale_obs=True,
    scale_obs=False,
):
    assert "NoFrameskip" in env_id
    env = gym.make(env_id)
    return AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=grayscale_obs,
        scale_obs=scale_obs,
    )


def time_env(model, env, num_iters=250):
    #  env.render()

    times = []
    model.initialize_experiences(env)

    for i in range(num_iters):
        t0 = time.time()
        done = False
        state = env.reset()
        t = 0
        while not done:
            tf_state = tf.convert_to_tensor(state, dtype=np.float32)
            tf_state = tf.expand_dims(tf_state, 0)
            action_qvals = model.q_net(tf_state)
            action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]

            _, _, done, _ = env.step(action)
            t += 1

            if t % 10 == 0:
                loss = model.optimize_model()

        t1 = time.time()
        times.append(t1 - t0)

        print(f"Episode length: {t}\tTime: {t1-t0}")

        if i % 50 == 0:
            print(f"Sum: {sum(times)}, Len: {len(times)}")
            print(f"Current average: {sum(times) / len(times)}")

    print(f"Average over {num_iters} attempts: {sum(times) / len(times)}")


name = "SpaceInvaders"
env_id = f"{name}NoFrameskip-v4"
#  env = make_atari_model(env_id, terminal_on_life_loss=True)

#  print("Built-in Gym Preprocessor: ")
#  time_env(env)

#  print()

env = make_atari_model(env_id)
model = Agent(env.action_space.n)

print("Local Atari/DeepMind Preprocessor: ")
time_env(model, env)
