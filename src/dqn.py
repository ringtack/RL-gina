import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

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


###################################################
#####             DDQN MODEL                  #####
###################################################
class DDQN(Model):
    def __init__(self, num_actions):
        """
        A classic implementation of the Nature DDQN for visual reinforcement learning tasks. We
        base our implementation off of
            https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        However, we take a Dueling DDQN approach, outlined in
            https://arxiv.org/abs/1511.06581
        and make minor modifications, doubling the dimension of the advantage/value layers
        by adding another convolutional layer.

        :param num_actions: number of actions in an environment
        """
        super().__init__()

        self.hidden = 1024
        self.num_actions = num_actions

        self.convs = Sequential(
            [
                Conv2D(filters=32, kernel_size=8, strides=4, activation="relu"),
                Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"),
                Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"),
                Conv2D(
                    filters=self.hidden, kernel_size=7, strides=1, activation="relu"
                ),
            ]
        )

        self.advantage_net = Dense(self.num_actions, input_shape=(self.hidden / 2,))
        self.value_net = Dense(1, input_shape=(self.hidden / 2,))

    def call(self, states):
        """
        Given input states (preferably stacked frames), performs a forward pass to estimate
        the Q-values of each action.

        :param states: a [batch_size, state_size] array of states
        :return: A [batch_size, num_actions] matrix representing the Q-values of each action
        """
        # Convolve the input state, then split into dueling components
        filters = self.convs(states)
        value_units, advantage_units = tf.split(filters, num_or_size_splits=2, axis=3)

        # flatten into (batch_size, hidden_size/2)
        #  value_units = tf.squeeze(value_units)
        #  advantage_units = tf.squeeze(advantage_units)
        value_units = tf.reshape(value_units, shape=(-1, 512))
        advantage_units = tf.reshape(advantage_units, shape=(-1, 512))

        # compute advantages and values
        adv = self.advantage_net(advantage_units)
        val = self.value_net(value_units)

        # compute Q value from advantages and values
        q_vals = val + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
        return q_vals


###################################################
#####               AGENT                     #####
###################################################


def epsilon(t):
    """
    Computes the epsilon value at a certain time step, following the linearly annealing
    epsilon-greedy strategy.
    """
    eps_frac = (EPSILON_END - EPSILON_START) / EPSILON_STEPS
    return max(EPSILON_END, EPSILON_START + t * eps_frac)


class Agent:
    def __init__(self, env, num_actions):
        self.env = env
        self.num_actions = num_actions

        self.q_net = DDQN(num_actions)
        self.target_net = DDQN(num_actions)
        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.experience = ExperienceBuffer(BUFFER_SIZE)

    def remember(self, *args):
        self.experience.remember(*args)

    def act(self, state, t):
        """
        Follow epsilon-greedy exploration to balance exploration-exploitation tradeoff
        """
        eps = epsilon(t)
        sample = random.random()

        if sample < eps:
            # Exploration
            action = self.env.action_space.sample()
        else:
            # Exploitation
            tf_state = tf.convert_to_tensor(state, dtype=np.float32)
            tf_state = tf.expand_dims(tf_state, 0)
            action_qvals = self.q_net(tf_state)
            action = tf.cast(tf.math.argmax(action_qvals, 1), tf.int32).numpy()[0]

        return action

    def initialize_experiences(self, env):
        """
        Initialize experience buffer with BUFFER_SIZE number of random experiences.
        """
        state = env.reset()
        done = False

        for _ in range(BUFFER_SIZE):
            # randomly initialize replay memory to capacity N
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            self.remember(state, action, reward, next_state, done)

            state = env.reset() if done else next_state

        print("Experience buffer initialized...")

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

        #  return loss
