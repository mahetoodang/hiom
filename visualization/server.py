from mesa.visualization.modules import NetworkModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from matplotlib import cm, colors

import sys
sys.path.append('../')

from src.Model import HIOM


# Setting model parameters
# decimal values can't be used so currently
# it is just good for understanding the general dynamics
# and looking at the network topology, but not for
# running actual experiments
model_params = {
    "attention_delta": UserSettableParameter('slider', 'Attention decay', 1, 1, 10),
    "persuasion": UserSettableParameter('slider', 'Persuasion', 1, 1, 10),
    "dt": UserSettableParameter('slider', "Time-step length", 1, 1, 10)
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
            "tooltip": "id: {} | A: {} | I: {} | O: {}".format(
                agents[0].unique_id,
                round(agents[0].attention, 2),
                round(agents[0].information, 2),
                round(agents[0].opinion, 2)
            )
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
