from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    opinion_vs_info_gif, \
    plot_opinion_stat_over_time
from scenarios.test import agents
from src.stats import compute_fractions_size


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
    plot_opinion_stat_over_time(
        "persuasion",
        [1, 10],
        compute_fractions_size,
        [{"dt": 0.1, "network_params": {"method": "er", "m": 2, "p": 0.1}},
        {"dt": 0.1, "network_params": {"method": "ba", "m": 2, "p": 0.1}}],
        N=1
    )

    opinion_vs_info_gif(opinion, information, attention)
