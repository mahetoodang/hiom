from src.Model import HIOM
from src.plotter import plot_opinion_distribution, \
    plot_single_opinion, \
    plot_single_information, \
    plot_single_attention, \
    opinion_vs_info
from scenarios.test import agents


if __name__ == "__main__":
    model = HIOM(agents, persuasion=0.1, attention_delta=0.05, r_min=0)
    model.run_model(2000)

    opinion = model.data_collector.get_model_vars_dataframe()["Opinion"]
    # plot_opinion_distribution(opinion)
    # plot_single_opinion(opinion, 8)
    attention = model.data_collector.get_model_vars_dataframe()["Attention"]
    # plot_single_attention(attention, 8)
    information = model.data_collector.get_model_vars_dataframe()["Information"]
    # plot_single_information(information, 8)
    # plot_opinion_distribution(opinion)
    # opinion_vs_info(opinion, information, attention)
