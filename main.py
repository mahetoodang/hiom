from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    opinion_vs_info_gif, test_opinion_stat_change, \
    plot_opinion_stat_over_time, plot_final_stats_over_time
from scenarios.test import agents
from src.stats import compute_fractions_size, compute_hartigan_opinions


if __name__ == "__main__":
    # Single model run
    model = HIOM(agents, persuasion=0.001, network_params={"method": "er", "p": 0.5})
    model.run_model()
    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    information = model.data_collector.get_model_vars_dataframe()["Information"]
    plot_opinion_distribution(opinion)

    # Plotting selected opinion statistics using various model parameters
    model_params = {"dt": 0.1, "network_params": {"method": "er", "m": 2, "p": 0.1}}
    test_opinion_stat_change("persuasion", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], stat_function=compute_fractions_size, N=5, 
                            xscale="log", model_params=model_params, ylabel="Fraction supporting minority")
    test_opinion_stat_change("persuasion", [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], stat_function=compute_hartigan_opinions, N=5, 
                            xscale="log", model_params=model_params, ylabel="Hartigan's D statistic")
    plot_opinion_stat_over_time(
        "persuasion",
        [1, 10],
        compute_fractions_size,
        [{"dt": 0.1, "network_params": {"method": "er", "m": 2, "p": 0.1}},
        {"dt": 0.1, "network_params": {"method": "ba", "m": 2, "p": 0.1}}],
        N=1
    )
    model_params_B = {"dt": 0.01,
                    "attention_delta": 0.1,
                    "a_min": -0.5,
                    "r_min": 0.5,
                    "sd_opinion": 0.15,
                    "sd_info": 0.005,
                    "network_params": {"method": "er", "p": 0.5}}
    plot_final_stats_over_time(model_params_B, persuasionH=1, persuasionL=0.1, total_time=5000)


    opinion_vs_info_gif(opinion, information, attention)
