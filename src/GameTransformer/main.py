import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
import tensorflow as tf
import tensorflow.keras as keras
from NaiveStateShift import *
from matplotlib.animation import FuncAnimation, PillowWriter
from DatasetGeneration import *
from GameTransformer import *

space_env = AtariPreprocessing(gym.make("SpaceInvaders-v0"), frame_skip=1, grayscale_obs=True, screen_size=84)
demon_env = AtariPreprocessing(gym.make("DemonAttack-v0"), frame_skip=1, grayscale_obs=True, screen_size=84)

SCREEN_SHAPE = [84, 84]
FLATTEN_SCREEN = SCREEN_SHAPE[0] * SCREEN_SHAPE[1]

class GraphableLoss(keras.losses.Loss):
    def __init__(self, name=None):
        super(GraphableLoss, self).__init__(name=name)
        self.losses = []
        self.batch_losses = []

    def call(self, y_true, y_pred):
        loss = self.compute_loss(y_true, y_pred)
        self.losses += loss.numpy().tolist()
        self.batch_losses.append(np.average(loss.numpy()))
        return loss

    def reset_losses(self):
        self.losses = []

    def graph_losses(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Losses During Training')

        ax1.plot(self.losses[1000:])
        ax1.set_title("Losses per Step")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")

        ax2.plot(self.batch_losses[100:])
        ax2.set_title("Losses per Batch")
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Loss")

        return fig


def animate_game(game_tensors, ax):
    game_numpy = game_tensors.numpy()
    if game_numpy.shape[1] > FLATTEN_SCREEN:
        game_numpy = game_numpy[:, :(FLATTEN_SCREEN - game_numpy.shape[1])]
    game_numpy = game_numpy.reshape([-1] + SCREEN_SHAPE)
    game_numpy = np.floor(game_numpy * 255)

    def animate(i): ax.imshow(game_numpy[i], cmap='gray')

    return animate


def compare_games(model, gen_set, filepath):
    sample_space, sample_demon = gen_set
    pred_demon = tf.convert_to_tensor(model.predict(sample_space))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    anim_space = animate_game(sample_space, ax1)
    anim_pred_demon = animate_game(pred_demon, ax2)
    anim_demon = animate_game(sample_demon, ax3)

    ax1.set_title("Space Invaders")
    ax2.set_title("Predicted Demon Attack")
    ax3.set_title("Demon Attack")

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    def animate(i):
        anim_space(i)
        anim_pred_demon(i)
        anim_demon(i)

    anim = FuncAnimation(fig, animate, frames=sample_space.shape[0], interval=1)
    anim.save(filepath, writer=PillowWriter(fps=24))


def train_naive(delta_method=False, time_dependent=False):
    EPISODES = 50

    assert delta_method or not time_dependent  # Time dependent only supported for delta methods

    def get_folder():
        if delta_method:
            if time_dependent:
                return "naive_time"
            return "naive_delta"
        else:
            return "naive"

    def get_data(episodes, length):
        if delta_method:
            if time_dependent:
                return populate_delta_sets(space_env, demon_env, episodes, length, True)
            return populate_delta_sets(space_env, demon_env, episodes, length)
        else:
            return populate_sets(space_env, demon_env, episodes, length)

    def get_model():
        if time_dependent:
            return create_naive_model(FLATTEN_SCREEN + 1, FLATTEN_SCREEN)
        return create_naive_model(FLATTEN_SCREEN)

    train_x, train_y = get_data(EPISODES, 10000)
    losses, model = get_model()

    model.fit(x=train_x, y=train_y, batch_size=64, epochs=1)
    print(f"{get_folder()}/chkpt/weights")

    fig = losses.graph_losses()
    fig.savefig(f"{get_folder()}/losses.png")

    gif_dataset = get_data(1, 100)
    compare_games(model, gif_dataset, f"{get_folder()}/games.gif")

def train_encoder():
    NUM_STEPS = 100
    train_space, train_demon = populate_experience_dataset(space_env, demon_env, NUM_STEPS)
    experience_model = GameEncoder(FLATTEN_SCREEN, 32, 16, 6)
    #
    print("model created")
    experience_model.train(train_space, train_demon, 1)
    experience_model.save(f"encoder/chkpt")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_encoder()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
