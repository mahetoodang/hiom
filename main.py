import numpy as np
import matplotlib.pyplot as plt

from src.Model import HIOM


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


if __name__ == "__main__":
    model = HIOM()
    model.run_model()

    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    plot_single_opinion(opinion, 8)
    attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    plot_single_attention(attention, 8)
    information = model.data_collector.get_model_vars_dataframe()["Information"]
    plot_single_information(information, 8)
