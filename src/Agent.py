from mesa import Agent as MesaAgent
import numpy as np


class Agent(MesaAgent):

    def __init__(self, model, unique_id, graph_id, neighbours):
        super().__init__(unique_id, model)
        self.graph_id = graph_id
        self.opinion = -1 + 2 * np.random.random()
        self.attention = np.random.random()
        self.information = -1 + 2 * np.random.random()
        self.neighbours = neighbours

    def step(self):
        if self.unique_id == self.model.active_agent:
            chosen_id = np.random.choice(self.neighbours)
            chosen_neighbour = None
            for agent in self.model.schedule.agents:
                if chosen_id == agent.graph_id:
                    chosen_neighbour = agent
            self.interact(chosen_neighbour)
        self.update_attention()
        self.update_opinion()

    def interact(self, chosen_neighbour):
        self.increase_attention()
        chosen_neighbour.increase_attention()
        chosen_neighbour.update_information(self)

    def update_opinion(self):
        d_opinion = - (
                np.power(self.opinion, 3) -
                (self.attention - self.model.a_min) * self.opinion -
                self.information
        ) * self.model.dt + np.random.normal(0, self.model.sd_opinion) * self.model.dt
        self.opinion += d_opinion

    def update_information(self, neighbour):
        expo = np.exp(-self.model.persuasion * (self.attention - neighbour.attention))
        frac = (1 - self.model.r_min) / (1 + expo)
        r = self.model.r_min + frac
        self.information = r * self.information \
            + (1-r) * neighbour.information \
            + np.random.normal(0, self.model.sd_info)

    def increase_attention(self):
        d_attention = self.model.attention_delta * (2 - self.attention)
        self.attention += d_attention

    def update_attention(self):
        frac = self.model.attention_delta / self.model.population # np.power(self.model.population, 2)
        d_attention = - 2 * frac * self.attention
        self.attention += d_attention
