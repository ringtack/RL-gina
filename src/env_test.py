import time

import gym
import numpy as np
from gym.wrappers.atari_preprocessing import AtariPreprocessing

from atari_wrappers import make_atari, wrap_deepmind


def make_atari_model(
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


def time_env(env, num_iters=250):
    #  env.render()

    times = []

    for i in range(num_iters):
        t0 = time.time()
        done = False
        env.reset()
        while not done:
            #  env.render()
            _, _, done, _ = env.step(env.action_space.sample())

        t1 = time.time()
        times.append(t1 - t0)

        if i % 50 == 0:
            print(f"Sum: {sum(times)}, Len: {len(times)}")
            print(f"Current average: {sum(times) / len(times)}")

    print(f"Average over {num_iters} attempts: {sum(times) / len(times)}")


name = "SpaceInvaders"
env_id = f"{name}NoFrameskip-v4"
env = make_atari_model(env_id, terminal_on_life_loss=True)

print("Built-in Gym Preprocessor: ")
time_env(env)

print()

env = make_atari(env_id)
env = wrap_deepmind(env)

print("Local Atari/DeepMind Preprocessor: ")
time_env(env)
