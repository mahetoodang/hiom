import random
import numpy as np
import networkx as nx
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from .Agent import Agent


class HIOM(Model):
    def __init__(
            self,
            population=400,
            dt=0.1,
            attention_delta=0.2,
            persuasion=1,
            a_min=-0.5,
            r_min=0.05,
            sd_opinion=0.15,
            sd_info=0.005
    ):

        super().__init__()

        self.population = population
        self.dt = dt
        self.attention_delta = attention_delta
        self.persuasion = persuasion
        self.a_min = a_min
        self.r_min = r_min
        self.sd_opinion = sd_opinion
        self.sd_info = sd_info

        # Initialize a scheduler
        self.schedule = BaseScheduler(self)

        # create the population
        self.M = nx.Graph()
        self.init_population()

        # agent who will interact this turn
        self.active_agent = None

        # add a schedule and a grid
        self.grid = NetworkGrid(self.M)

        # add datacollector
        self.data_collector = DataCollector({
            "Opinion": lambda m: self.collect_opinions(),
            "Attention": lambda m: self.collect_attentions(),
            "Information": lambda m: self.collect_informations()
        })

        # this is required for the data_collector to work
        self.running = True
        self.data_collector.collect(self)

    def init_population(self):
        self.M = nx.fast_gnp_random_graph(self.population, 0.1)
        for node in self.M.nodes:
            neighbours = [edge[1] for edge in self.M.edges(node)]
            self.new_agent(node, neighbours)

    def new_agent(self, graph_id, neighbours):
        agent_id = self.next_id()
        agent = Agent(self, agent_id, graph_id, neighbours)
        self.schedule.add(agent)

    def step(self):
        '''
        Execute next time step.
        '''
        self.choose_agent()
        self.schedule.step()
        # Save the statistics
        self.data_collector.collect(self)

    def choose_agent(self):
        ids = []
        attentions = []
        attention_sum = 0
        for agent in self.schedule.agents:
            ids.append(agent.unique_id)
            attentions.append(agent.attention)
            attention_sum += agent.attention
        attentions = np.array(attentions) / attention_sum
        self.active_agent = random.choices(ids, attentions, k=1)[0]

    def collect_opinions(self):
        opinions = []
        for agent in self.schedule.agents:
            opinions.append([agent.unique_id, agent.opinion])
        return opinions

    def collect_attentions(self):
        attentions = []
        for agent in self.schedule.agents:
            attentions.append([agent.unique_id, agent.attention])
        return attentions

    def collect_informations(self):
        informations = []
        for agent in self.schedule.agents:
            informations.append([agent.unique_id, agent.information])
        return informations

    def run_model(self, step_count=500):
        '''
        Runs model.
        '''
        for i in range(step_count):
            self.step()
