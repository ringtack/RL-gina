import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from main import GraphableLoss


def create_naive_model(inp_size, out_size=0):
    outsize = out_size if out_size > 0 else inp_size
    model = keras.Sequential([
        layers.Dense(inp_size, activation='relu', kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(inp_size * 1.5, activation='relu', kernel_initializer="he_normal"),
        layers.Dropout(0.2),
        layers.Dense(inp_size * 1.5, activation='relu', kernel_initializer="he_normal"),
        layers.Dropout(0.2),
        layers.Dense(out_size)
    ])
    losses = NaiveLoss()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=losses, run_eagerly=True)
    return losses, model

class NaiveLoss(GraphableLoss):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_loss(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
        return mse
