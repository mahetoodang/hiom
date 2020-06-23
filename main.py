from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    plot_opinion_distribution_animation
    # plot_single_opinion, \
    # plot_single_information, \
    # plot_single_attention, \
from scenarios.test import agents
from src.stats import compute_hartigan


if __name__ == "__main__":
    model = HIOM(agents, network_params={"method": "er", "p": 0.1})
    model.run_model()

    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    # plot_opinion_distribution(opinion)
    # plot_single_opinion(opinion, 8)
    # plot_opinion_distribution_animation(opinion, 500)

    # Hardigan Dip Test
    compute_hartigan(opinion[500])
    
    # attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    # plot_single_attention(attention, 8)
    # information = model.data_collector.get_model_vars_dataframe()["Information"]
    # plot_single_information(information, 8)
