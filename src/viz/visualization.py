import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv


def strip_tensorflow(filename):
    with open(filename, "r") as file:
        all = file.read()
        all_stripped = all.replace('"tf.Tensor(', '').replace(', shape=(), dtype=float32)"', '')

    with open(filename, "w") as file:
        file.write(all_stripped)


def graph_losses(ax, loss_file, groups=0, logscale=True):
    losses = np.genfromtxt(loss_file, delimiter=",", dtype=np.float32)[1:]

    if logscale:
        ax.set_yscale('log')

    if groups > 0:
        sns.regplot(losses[:, 0], losses[:, 1], color='g', x_bins=groups, ax=ax)
        ax.set_title("Grouped Losses with Line of Best Fit")
    else:
        ax.plot(losses[:, 0], losses[:, 1])
        ax.set_title("Losses")

    ax.set_xlabel("Step")
    ax.set_ylabel(f"{'Log ' if logscale else ''}Losses")


def graph_q(ax, q_file, groups=0, logscale=False):
    qs = np.genfromtxt(q_file, delimiter=",", dtype=np.float32)[1:]

    if logscale:
        ax.set_yscale('log')

    if groups > 0:
        sns.regplot(list(range(1, 1 + len(qs[:, 1]))), qs[:, 1], color='orange', x_bins=groups, ax=ax)
        ax.set_title("Grouped Q with Line of Best Fit")
    else:
        ax.plot(list(range(1, 1 + len(qs[:, 1]))), qs[:, 1], color="red")
        ax.set_title("Q")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{'Log ' if logscale else ''}Q")


def graph_rewards(ax, reward_file, groups=0, logscale=False):
    rewards = np.genfromtxt(reward_file, delimiter=",", dtype=np.float32)

    if logscale:
        ax.set_yscale('log')

    if groups > 0:
        sns.regplot(list(range(1, 1 + len(rewards[0]))), rewards[0], color='purple', x_bins=groups, ax=ax)
        ax.set_title("Grouped Rewards with Line of Best Fit")
    else:
        ax.plot(rewards[0], color="teal")
        ax.set_title("Rewards")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{'Log ' if logscale else ''}Reward")


def append_rewards(ax, files, groups=0, logscale=False):
    file_list = [numpy.genfromtxt(file, delimiter=",", dtype=np.float32) for file in files]
    for file in file_list:
        print(file.shape)
    rewards = numpy.concatenate(file_list, axis=1)

    if logscale:
        ax.set_yscale('log')

    if groups > 0:
        sns.regplot(list(range(1, 1 + len(rewards[0]))), rewards[0], color='purple', x_bins=groups, ax=ax)
        ax.set_title("Grouped Rewards with Line of Best Fit")
    else:
        ax.plot(rewards[0], color="teal")
        ax.set_title("Rewards")

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"{'Log ' if logscale else ''}Reward")


if __name__ == '__main__':
    sns.set_theme()
    # fig1, ax1 = plt.subplots(2)
    # graph_losses(ax1[0], "joint-da/td1-loss.csv")
    # graph_losses(ax1[1], "joint-da/td1-loss.csv", groups=50)
    # fig1.suptitle("Losses for Joint Demon Attack")
    #
    # fig2, ax2 = plt.subplots(2)
    # graph_losses(ax2[0], "joint-si/td2-loss.csv")
    # graph_losses(ax2[1], "joint-si/td2-loss.csv", groups=50)
    # fig2.suptitle("Losses for Joint Space Invaders")
    #
    # fig3, ax3 = plt.subplots(2)
    # graph_q(ax3[0], "joint-si/qvals.csv")
    # graph_q(ax3[1], "joint-si/qvals.csv", groups=50)
    # fig3.suptitle("Q for Joint Space Invaders")
    #
    # fig4, ax4 = plt.subplots(2)
    # graph_q(ax4[0], "joint-da/qvals.csv")
    # graph_q(ax4[1], "joint-da/qvals.csv", groups=50)
    # fig4.suptitle("Q for Joint Demon Attack")

    fig5, ax5 = plt.subplots(2)
    append_rewards(ax5[0], ["../rwds/joint-da/reward1-12-09-23-21.csv","../rwds/joint-da/reward1-12-10-00-45.csv",
                            "../rwds/joint-da/reward1-12-10-01-00.csv", "../rwds/joint-da/reward1-12-10-01-12.csv",
                            "../rwds/joint-da/reward1-12-10-01-24.csv", "../rwds/joint-da/reward1-12-10-01-36.csv",
                            "../rwds/joint-da/reward1-12-10-04-52.csv"])

    append_rewards(ax5[1], ["../rwds/joint-da/reward1-12-09-23-21.csv",  "../rwds/joint-da/reward1-12-10-00-45.csv",
                            "../rwds/joint-da/reward1-12-10-01-00.csv", "../rwds/joint-da/reward1-12-10-01-12.csv",
                            "../rwds/joint-da/reward1-12-10-01-24.csv", "../rwds/joint-da/reward1-12-10-01-36.csv",
                            "../rwds/joint-da/reward1-12-10-04-52.csv"], groups=50)
    fig5.suptitle("Rewards for Joint Demon Attack")

    fig6, ax6 = plt.subplots(2)
    append_rewards(ax6[0], ["../rwds/joint-si/reward2-12-09-23-21.csv",
                            "../rwds/joint-si/reward2-12-10-00-45.csv", "../rwds/joint-si/reward2-12-10-01-00.csv",
                            "../rwds/joint-si/reward2-12-10-01-12.csv", "../rwds/joint-si/reward2-12-10-01-24.csv",
                            "../rwds/joint-si/reward2-12-10-01-36.csv",
                            "../rwds/joint-si/reward2-12-10-04-52.csv"])

    append_rewards(ax6[1], ["../rwds/joint-si/reward2-12-09-23-21.csv",
                            "../rwds/joint-si/reward2-12-10-00-45.csv", "../rwds/joint-si/reward2-12-10-01-00.csv",
                            "../rwds/joint-si/reward2-12-10-01-12.csv", "../rwds/joint-si/reward2-12-10-01-24.csv",
                            "../rwds/joint-si/reward2-12-10-01-36.csv",
                            "../rwds/joint-si/reward2-12-10-04-52.csv"], groups=50)
    fig6.suptitle("Rewards for Joint Space Invaders")

    # fig7, ax7 = plt.subplots(2)
    # graph_losses(ax7[0], "shared/td-loss-12-10-00-16.csv", logscale=True)
    # graph_losses(ax7[1], "shared/td-loss-12-10-00-16.csv", groups=50, logscale=True)
    # fig7.suptitle("Losses for Shared Model")
    #
    # fig8, ax8 = plt.subplots(2)
    # graph_q(ax8[0], "shared/qvals-12-10-00-16.csv", logscale=True)
    # graph_q(ax8[1], "shared/qvals-12-10-00-16.csv", groups=50, logscale=True)
    # fig8.suptitle("Q for Shared Model")
    #
    # fig9, ax9 = plt.subplots(2)
    # graph_rewards(ax9[0], "../rwds/shared/reward-12-10-00-16.csv")
    # graph_rewards(ax9[1], "../rwds/shared/reward-12-10-00-16.csv", groups=50)
    # fig9.suptitle("Q for Shared Model")
    #
    # fig10, ax10 = plt.subplots(2)
    # graph_losses(ax10[0], "base-da/td-loss-12-09-17-14.csv", logscale=True)
    # graph_losses(ax10[1], "base-da/td-loss-12-09-17-14.csv", groups=50, logscale=True)
    # fig10.suptitle("Losses for Base DA Model")
    #
    # fig11, ax11 = plt.subplots(2)
    # graph_q(ax11[0], "base-da/qvals-12-09-17-14.csv", logscale=True)
    # graph_q(ax11[1], "base-da/qvals-12-09-17-14.csv", groups=50, logscale=True)
    # fig11.suptitle("Q for Base DA Model")
    #
    # fig12, ax12 = plt.subplots(2)
    # graph_rewards(ax12[0], "../rwds/base-da/reward-12-09-17-14.csv")
    # graph_rewards(ax12[1], "../rwds/base-da/reward-12-09-17-14.csv", groups=50)
    # fig12.suptitle("Rewards for Base DA Model")

    plt.show()
