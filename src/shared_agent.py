import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop

from dqn import DDQN
from experience_buffer import ExperienceBuffer, Transition
from settings import (
    BATCH_SIZE,
    BUFFER_SIZE,
    EPSILON_END,
    EPSILON_START,
    EPSILON_STEPS,
    GAMMA,
    LEARNING_RATE,
    NUM_EPISODES,
)


def epsilon(t):
    """
    Computes the epsilon value at a certain time step, following the linearly annealing
    epsilon-greedy strategy.
    """
    eps_frac = (EPSILON_END - EPSILON_START) / EPSILON_STEPS
    return max(EPSILON_END, EPSILON_START + t * eps_frac)


class SharedAgent:
    def __init__(self, env1, env2, stack):
        self.env1 = env1
        self.env2 = env2

        assert env1.action_space.n == env2.action_space.n
        self.num_actions = env1.action_space.n

        self.q_net = DDQN(self.num_actions, stack)
        self.target_net = DDQN(self.num_actions, stack)
        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.experience = ExperienceBuffer(2 * BUFFER_SIZE)

    def remember(self, *args):
        self.experience.remember(*args)

    def initialize_experiences(self):
        state1 = self.env1.reset()
        done1 = False
        state2 = self.env2.reset()
        done2 = False

        for _ in range(BUFFER_SIZE):
            # randomly initialize replay memory to capacity N with both games
            action1 = self.env1.action_space.sample()
            next_state1, reward1, done1, _ = self.env1.step(action1)
            self.remember(state1, action1, reward1, next_state1, done1)
            state1 = self.env1.reset() if done1 else next_state1

            action2 = self.env2.action_space.sample()
            next_state2, reward2, done2, _ = self.env2.step(action2)
            self.remember(state2, action2, reward2, next_state2, done2)
            state2 = self.env2.reset() if done2 else next_state2
        print("Shared experience buffer initialized...")

    def act(self, state1, state2, t):
        eps = epsilon(t)
        sample = random.random()

        if sample < eps:
            # Exploration
            action1 = self.env1.action_space.sample()
        else:
            # Exploitation
            tf_state = tf.convert_to_tensor(state1, dtype=np.float32)
            tf_state = tf.expand_dims(tf_state, 0)
            action_qvals = self.q_net(tf_state)
            action1 = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]

        sample = random.random()
        if sample < eps:
            action2 = self.env2.action_space.sample()
        else:
            tf_state = tf.convert_to_tensor(state2, dtype=np.float32)
            tf_state = tf.expand_dims(tf_state, 0)
            action_qvals = self.q_net(tf_state)
            action2 = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]

        return action1, action2

    def update_target_net(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def optimize_model(self):
        # Fetch batched memories from experience buffer
        transitions = self.experience.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Get components of batches
        state_batch = tf.convert_to_tensor(batch.state, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(batch.action, dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(batch.next_state, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(batch.done, dtype=tf.float32)

        # Find the best possible action in the next states
        max_next_actions = tf.cast(
            tf.math.argmax(self.q_net(next_state_batch), 1), tf.int32
        )

        # assemble actions into [batch index, action]
        action_range = tf.range(action_batch.shape[0], dtype=tf.int32)
        indexed_actions = tf.stack([action_range, max_next_actions], axis=1)

        # Compute the q values associated with each action given the next state, then
        # use the best possible actions in the next state to get the target q values
        target_q_vals = self.target_net(next_state_batch)
        indexed_q_vals = tf.gather_nd(target_q_vals, indexed_actions)
        target_q = reward_batch + GAMMA * indexed_q_vals * (1.0 - done_batch)

        # Compute the q values of the performed action
        action_q_val = self.q_net(state_batch)
        net_q = tf.reduce_sum(
            (
                action_q_val
                * tf.one_hot(action_batch, self.num_actions, dtype=tf.float32)
            ),
            axis=1,
        )

        # Rudimentary research [TODO: find some paper] suggests that Huber loss performs
        # better in error clipping
        #  loss = tf.reduce_mean(tf.square(target_q - net_q))
        #  def loss(y_true, y_pred):
        #  """Final loss construction to be compiled"""
        return tf.reduce_mean(tf.keras.losses.Huber()(target_q, net_q))
