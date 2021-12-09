import argparse
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
    n_env = gym.make(env_id)
    return AtariPreprocessing(n_env)


def time_env(model, env, num_iters=10):
    #  env.render()
    times = []
    model.initialize_experiences(env)

    for i in range(num_iters):
        frames = []

        t0 = time.time()

        done = False
        state = env.reset()
        t = 0
        while not done:
            frames.append(Image.fromarray(env.render(mode="rgb_array")))
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

        with open("./gifs/test.gif", "wb+") as f:
            im = frames[0]
            im.save(
                f, save_all=True, optimize=True, duration=30, append_images=frames[1:]
            )

        t2 = time.time()
        print(f"Time to write gif: {t2 - t1}")

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
model = Agent(env, env.action_space.n)

print("Local Atari/DeepMind Preprocessor: ")
time_env(model, env)


#  frames = []
#  done = False

#  state = env.reset()

#  while not done:
#  frames.append(Image.fromarray(env.render(mode="rgb_array")))
#  _, _, done, _ = env.step(env.action_space.sample())

#  with open("./gifs/test.gif", "wb+") as f:
#  im = frames[0]
#  im.save(f, save_all=True, append_images=frames[1:])
