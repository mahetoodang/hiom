import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors


def plot_opinion_distribution(opinions):
    iterations = len(opinions)
    steps = [0, int(iterations * 1/3), int(iterations * 2/3), iterations-1]

    fig = plt.figure()
    fig.suptitle("Polarization of opinions")

    for i, step in enumerate(steps):
        ops = [ag[1] for ag in opinions[step]]
        plt.subplot(2, 2, i+1)
        plt.hist(ops, color="b")
        plt.title("Step " + str(step))

    plt.tight_layout()
    plt.show()


def opinion_vs_info(opinions, informations, attentions):
    iterations = len(opinions)
    # steps = [0, int(iterations * 1 / 3), int(iterations * 2 / 3), iterations - 1]
    steps = [0, 20, 40, 150]

    fig = plt.figure()

    for i, step in enumerate(steps):
        ops = [ag[1] for ag in opinions[step]]
        infs = [ag[1] for ag in informations[step]]
        atts = [ag[1] for ag in attentions[step]]
        a_mean = np.mean(atts)
        plt.subplot(2, 2, i + 1)
        plt.scatter(infs, ops, c="b", alpha=0.5)
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.75, 1.75))
        plt.title("Step " + str(step) + ", E(A)=" + str(np.round(a_mean, 2)))

    plt.tight_layout()
    plt.show()


def opinion_vs_info_gif(opinions, informations, attentions):
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
    scat = ax.scatter([], [], c="b", alpha=0.5)
    c_map = cm.get_cmap('jet', 255)
    legend = fig.colorbar(scat, )
    legend.set_label('Attention', labelpad=-20, y=1.1, rotation=0)

    def init():
        ax.set_xlabel("Information")
        ax.set_ylabel("Opinion")
        return scat,

    def update(frame):
        ops = [ag[1] for ag in opinions[frame]]
        infs = [ag[1] for ag in informations[frame]]
        atts = [colors.rgb2hex(c_map(ag[1] / 2)) for ag in attentions[frame]]
        data = [[infs[i], ops[i]] for i in range(len(ops))]
        scat.set_offsets(data)
        scat.set_edgecolors(atts)
        scat.set_facecolors(atts)
        return scat,

    anim = FuncAnimation(fig, update, len(opinions), init_func=init, blit=True)
    anim.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()


def plot_single_opinion(opinions, id):
    x = np.linspace(0, 500, 501)
    y = [row[id-1][1] for row in opinions]
    plt.plot(x, y, 'b')
    plt.xlabel("Step")
    plt.ylabel("Opinion")
    plt.title("Opinion of agent " + str(id))
    plt.show()


def plot_single_attention(attentions, id):
    x = np.linspace(0, 500, 501)
    y = [row[id-1][1] for row in attentions]
    plt.plot(x, y, 'b')
    plt.xlabel("Step")
    plt.ylabel("Attention")
    plt.title("Attention of agent " + str(id))
    plt.show()


def plot_single_information(information, id):
    x = np.linspace(0, 500, 501)
    y = [row[id-1][1] for row in information]
    plt.plot(x, y, 'b')
    plt.xlabel("Step")
    plt.ylabel("Information")
    plt.title("Information of agent " + str(id))
    plt.show()
