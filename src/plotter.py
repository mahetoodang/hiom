import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm, colors

from src.Model import HIOM
from scenarios.test import agents


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
    # steps should be changed according to scenario
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


def plot_opinion_distribution_animation(opinions):
    # Animated plot of opinion distribution (iterating over steps)
    # To save the animation, uncomment last lines
    steps = len(opinions)

    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(0, 100))

    raw_opinions = [x[1] for x in opinions[0]]
    counts, bins, bars = plt.hist(raw_opinions)

    def animate(i, bars):
        plt.cla()
        plt.ylim(0,100)
        plt.xlabel("Opinion")
        plt.ylabel("Number of people")

        raw_opinions = [x[1] for x in opinions[i]]
        counts, bins, bars = plt.hist(raw_opinions, range=(-2, 2))
        
    plot_animation = animation.FuncAnimation(fig, animate, steps, fargs=[bars])
    plot_animation.save('animation.gif', writer='imagemagick', fps=60)
    # plt.show()


def plot_scatter(values, stdevs, labels=None, xlabel="", ylabel="", xscale="linear", yscale="linear"):
    # General function to simplify plotting of various statistics 
    if labels is None:
        plt.plot(range(len(values)), values)
    else:
        plt.errorbar(labels, values, yerr=stdevs)
    plt.xlabel(ylabel)
    plt.ylabel(xlabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid()
    plt.show()


def plot_opinion_stat_over_time(tested_parameter, tested_values, stat_function, model_params_list, N=1, total_time=300):
    # Function to plot how statistics of opinion change while varying selected parameter (tested_paramters)
    # over predefined set of values (tested_values) in a number of models defined in the list (model_params_list).
    # Number of test repetitions can be set via N variable, just like total time that is to be simulated.
    # It is important to remember that the number of iterations is equal to the (total_time)/persuasion,
    # since we assume that higher persuasion require more time.

    # To interpret the results, the number of the run has to computed.
    counter = 1
    for dv in tested_values:
        for _ in range(N):
            for model_params in model_params_list:
                model_params[tested_parameter] = dv
                model = HIOM(agents, **model_params)
                model.run_model(int(total_time/model.persuasion))

                opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
                iterations = len(opinion)
                time = [i * model.persuasion for i in range(iterations)]
                op_time = [None] * iterations
                for i in range(iterations):
                    op_time[i] = stat_function(opinion[i])[0]
                plt.plot(time, op_time, label = ("run #" + str(counter)))
                counter += 1
    plt.title("")
    plt.xlabel("Time")
    # plt.ylabel("Mean opinion")
    plt.ylabel("Mean opinion")
    plt.grid()
    plt.legend()
    plt.show()


def test_opinion_stat_change(tested_parameter, tested_values, stat_function, N=3, step_count=500, xscale="linear", yscale="linear", model_params={}, ylabel=""):
    last_opinions_avg = []
    last_opinions_stdev = []
    for dv in tested_values:
        model_params[tested_parameter] = dv
        results = []
        for _ in range(N):
            model = HIOM(agents, **model_params)
            model.run_model(step_count=step_count)
            opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
            results.append(stat_function(opinion[step_count])[0])
        last_opinions_avg.append(np.mean(results))
        last_opinions_stdev.append(np.std(results))
    plot_scatter(last_opinions_avg, last_opinions_stdev, ylabel=ylabel, xlabel=tested_parameter, labels=tested_values, xscale=xscale, yscale=yscale)

def plot_final_stats_over_time(model_params, persuasionL, persuasionH, N=1, total_time=5000):
    counter = 1
    for _ in range(N):
        model = HIOM(agents, **model_params)
        model.persuasion = persuasionL
                    # dt=0.01,
                    # attention_delta=0.1,
                    # persuasion= persuasionL,
                    # a_min=-0.5,
                    # r_min=0.5,
                    # sd_opinion=0.15,
                    # sd_info=0.005,
                    # network_params={"method": "er", "p": 0.5})
        model.run_model(int(total_time/persuasionL))

        model2 = HIOM(agents, **model_params)
        model2.persuasion = persuasionH
                    
        model2.run_model(int(total_time/persuasionH))

        opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
        iterations = len(opinion)
        time = [i * persuasionL for i in range(iterations)]
        op_time = [None] * iterations
        for i in range(iterations):
                ops = [ag[1] for ag in opinion[i]]
                op_time[i] = np.mean(ops)
                # op_time[i] = compute_fractions_size(opinion[i])[0]

        plt.plot(time, op_time, label = ("Low persuasion run #" + str(counter)))
        plt.xlabel("Time")
        # plt.ylabel("Mean opinion")
        plt.ylabel("Mean opinion")

        # plt.title("Fidelity Effect")

        opinion2 = model2.data_collector.get_model_vars_dataframe()["Opinion"]
        iterations = len(opinion2)
        time = [i * persuasionH for i in range(iterations)]
        op_time = [None] * iterations
        for i in range(iterations):
                ops = [ag[1] for ag in opinion2[i]]
                op_time[i] = np.mean(ops)
                # op_time[i] = compute_fractions_size(opinion2[i])[0]

        plt.plot(time, op_time, '--', label = ("High persuasion run #" + str(counter+1)))
        counter += 2
    plt.grid()
    plt.legend()

    plt.show()
