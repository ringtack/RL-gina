import pickle

import gym

from atari_wrappers import make_atari_model

env = make_atari_model("DemonAttackNoFrameskip-v4", frame_stack=True)

state = env.reset()
done = False

states = []

for _ in range(500):
    # randomly initialize replay memory to capacity N
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    states.append(state)

    state = env.reset() if done else next_state

with open("viz/q_states_da_stack.pickle", "wb+") as p_f:
    pickle.dump(states, p_f)
