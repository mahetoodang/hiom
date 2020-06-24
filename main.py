from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    plot_opinion_distribution_animation, \
    plot_scatter
    # plot_single_opinion, \
    # plot_single_information, \
    # plot_single_attention, \
from scenarios.test import agents
from src.stats import compute_hartigan_opinions, compute_fractions_size
import numpy as np

def test_stat_change(tested_parameter, tested_values, stat_function, N=3, step_count=500, xscale="linear", yscale="linear", model_params={}):
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
    plot_scatter(last_opinions_avg, last_opinions_stdev, xlabel=tested_parameter, labels=tested_values, xscale=xscale, yscale=yscale)

if __name__ == "__main__":
    model = HIOM(agents, network_params={"method": "er", "p": 0.1})
    model.run_model()

    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    plot_opinion_distribution(opinion)
    # plot_single_opinion(opinion, 8)
    # plot_opinion_distribution_animation(opinion, 500)

    # Hardigan Dip Test
    # print(compute_hartigan_opinions(opinion[500]))

    # Plotting selected statistics with varying model parameters
    # test_stat_change("dt", [0.0001, 0.001, 0.01, 0.1, 0.5], stat_function=compute_fractions_size, xscale="log")
   
    # attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    # plot_single_attention(attention, 8)
    # information = model.data_collector.get_model_vars_dataframe()["Information"]
    # plot_single_information(information, 8)
