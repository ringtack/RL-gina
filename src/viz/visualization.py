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
        sns.regplot(losses[:,0], losses[:,1], color='g', x_bins=groups, ax=ax)
        ax.set_title("Grouped Losses with Line of Best Fit")
    else:
        ax.plot(losses[:,0], losses[:,1])
        ax.set_title("Losses")

    ax.set_xlabel("Step")
    ax.set_ylabel(f"{'Log ' if logscale else ''}Losses")

    print(ax)

if __name__ == '__main__':
    sns.set_theme()
    fig1, ax1= plt.subplots(2)
    graph_losses(ax1[0], "base-si/LOSS_FINAL.csv")
    graph_losses(ax1[1], "base-si/LOSS_FINAL.csv", groups=50)
    fig1.suptitle("Losses for Base1")

    fig2, ax2=plt.subplots(2)
    graph_losses(ax2[0], "base-si2/td-loss-12-09-16-52.csv")
    graph_losses(ax2[1], "base-si2/td-loss-12-09-16-52.csv", groups=50)
    fig2.suptitle("Losses for Base2")
    plt.show()
