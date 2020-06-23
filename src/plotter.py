import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

def plot_opinion_distribution_animation(opinions, steps):
    # Animated plot of opinion distribution (iterating over steps)
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(0, 100))

    raw_opinions = [x[1] for x in opinions[0]]
    counts, bins, bars = plt.hist(raw_opinions)
    print(bars)

    def animate(i, bars):
        plt.cla()
        plt.ylim(0,100)
        raw_opinions = [x[1] for x in opinions[i]]
        counts, bins, bars = plt.hist(raw_opinions, range=(-2, 2))
        
    plot_animation = animation.FuncAnimation(fig, animate, steps, fargs=[bars])
    plt.show()