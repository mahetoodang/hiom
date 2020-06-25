from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    plot_opinion_distribution_animation, \
    plot_scatter
    plot_single_opinion, \
    plot_single_information, \
    plot_single_attention, \
    opinion_vs_info, \
    opinion_vs_info_gif
from scenarios.test import agents
from src.stats import compute_hartigan_opinions, compute_fractions_size, compute_mean_opinion
import numpy as np
import matplotlib.pyplot as plt

def test_opinion_stat_change(tested_parameter, tested_values, stat_function, N=3, step_count=500, xscale="linear", yscale="linear", model_params={}):
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

# def plot_opinion_stat_over_time(opinions, stat_func):
#     results = []
#     for opinion in opinions:
#         results.append(stat_func(opinion)[0])
#     plot_scatter(results, 0)

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


if __name__ == "__main__":

    # # Single model run
    model = HIOM(agents, persuasion=0.001, network_params={"method": "er", "p": 0.5})
    model.run_model()
    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    
    # plot_single_opinion(opinion, 8)
    plot_opinion_distribution(opinion)
    # plot_opinion_distribution_animation(opinion)

    # plot_opinion_stat_over_time(opinion, compute_mean_opinion)
    # plot_opinion_stat_over_time(opinion, compute_hartigan_opinions)
    # plot_opinion_stat_over_time(opinion, compute_fractions_size)
    
    # # Hardigan Dip Test
    # print(compute_hartigan_opinions(opinion[500]))

    # # Plotting selected opinion statistics using various model parameters
    model_params = {"dt": 0.1, "network_params": {"method": "er", "m": 2, "p": 0.1}}
    # model_params = {"dt": 0.1, "network_params": { "method": "social_media", "path": "data/facebook_combined.txt" } } # Some networks, like this one, require changes in scenarios/test.py (number of agents)
    # test_opinion_stat_change("dt", [0.0001, 0.001, 0.01, 0.1, 0.5], stat_function=compute_fractions_size, xscale="log", model_params=model_params)
    # test_opinion_stat_change("attention_delta", [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5], stat_function=compute_fractions_size, N=5, xscale="log", model_params=model_params)
    # test_opinion_stat_change("persuasion", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], stat_function=compute_fractions_size, N=5, xscale="log", model_params=model_params)
    # test_opinion_stat_change("persuasion", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], stat_function=compute_fractions_size, N=1, xscale="log", model_params=model_params)
    # test_opinion_stat_change("persuasion", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], stat_function=compute_fractions_size, N=5, xscale="log", model_params=model_params)
    # test_opinion_stat_change("r_min", [0.0001, 0.001, 0.01, 0.1, 0.5], stat_function=compute_fractions_size, xscale="log", model_params=model_params)
    plot_opinion_stat_over_time("persuasion", 
                                [1, 10], 
                                compute_fractions_size, 
                                [{"dt": 0.1, "network_params": {"method": "er", "m": 2, "p": 0.1}}, 
                                {"dt": 0.1, "network_params": {"method": "ba", "m": 2, "p": 0.1}}], 
                                N=1)
    # # Plot attention
    # attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    # plot_single_attention(attention, 8)

    # # Plot infromation
    # information = model.data_collector.get_model_vars_dataframe()["Information"]
    # plot_single_information(information, 8)
    opinion_vs_info_gif(opinion, information, attention)
