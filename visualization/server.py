from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
# from mesa.visualization.modules import ChartModule
from matplotlib import cm, colors

import sys
sys.path.append('../')

from src.Model import HIOM


# Setting model parameters
model_params = {
    # "population_size": UserSettableParameter('slider', 'Population size', 1, 1, 500)
    "attention_delta": UserSettableParameter('slider', 'Attention decay', 1, 1, 10)
}

# Setting color maps
agent_cmap = cm.get_cmap('jet', 255)


def network_portrayal(G):
    # The model ensures there is always 1 agent per node

    def node_color(agent):
        intensity = agent.opinion
        if intensity > 1:
            intensity = 1
        elif intensity < -1:
            intensity = -1
        intensity = (intensity + 1) / 2
        return colors.rgb2hex(agent_cmap(intensity))

    def edge_color(agent1, agent2):
        return "#e6e6e6"

    def edge_width(agent1, agent2):
        return 1

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            "tooltip": "id: {}".format(agents[0].unique_id),
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


# Drawing the grid, initialising elements and launching server
canvas_element = NetworkModule(network_portrayal, 500, 500, library="d3")
element_list = [canvas_element]

server = ModularServer(HIOM, element_list, "HIOM", model_params)

server.launch()
