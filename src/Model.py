import random
import numpy as np
from mesa import Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler

from .Agent import Agent
from .Network import Network
import sys
sys.path.append('../')

from scenarios.test import agents


class HIOM(Model):
    def __init__(
            self,
            agents=agents,
            dt=0.1,
            attention_delta=0.1,
            persuasion=1,
            a_min=-0.5,
            r_min=0.05,
            sd_opinion=0.15,
            sd_info=0.005,
            network_params=None
    ):

        super().__init__()

        self.dt = dt
        self.attention_delta = attention_delta
        self.persuasion = persuasion
        self.a_min = a_min
        self.r_min = r_min
        self.sd_opinion = sd_opinion
        self.sd_info = sd_info

        # initialize a scheduler
        self.schedule = BaseScheduler(self)

        # create the population
        self.population = 0
        if network_params is None:
            network_params = {"method": "er", "p": 0.1}
        self.init_population(agents, network_params)
        # self.G = nx.Graph()

        # generates network topology
        # not used right now, but can be used for mesa visualization
        self.grid = NetworkGrid(self.G)

        # create agents
        self.create_agents(agents)

        # agent who will interact this turn
        self.active_agent = None

        # add datacollector
        # collects opinion, information and attention each step
        self.data_collector = DataCollector({
            "Opinion": lambda m: self.collect_opinions(),
            "Attention": lambda m: self.collect_attentions(),
            "Information": lambda m: self.collect_informations()
        })

        # this is required for the data_collector to work
        self.running = True
        self.data_collector.collect(self)

    def init_population(self, agents, network_params):
        # population size is calculated and an array of
        # possible agent types is stored for pop generation
        pop_size = 0
        for atype in agents:
            pop_size += atype["n"]
        self.population = pop_size
        # network topology is initialized
        network = Network(n=self.population, params=network_params)
        self.G = network.get_graph()

    def create_agents(self, agents):
        types = []
        for atype in agents:
            types.append([atype["n"], atype["generator"]])
        # for each node in the network, agent type
        # is chosen by random choice and an agent is created
        for node in self.G.nodes:
            # finds all neighbours in the network
            neighbours = [edge[1] for edge in self.G.edges(node)]
            # chooses the agent type by random choice
            type_idx = random.choice(range(len(types)))
            agent_type = types[type_idx]
            # creates new agent using the generator func (agent_type[1])
            self.new_agent(node, neighbours, agent_type[1])
            # number of agents needed to create is reduced
            # and the type is removed if no more agents of this
            # type can be created
            agent_type[0] -= 1
            if agent_type[0] == 0:
                del types[type_idx]

    def new_agent(self, graph_id, neighbours, generator):
        agent_id = self.next_id()
        agent = Agent(
            self,
            agent_id,
            graph_id,
            neighbours,
            generator
        )
        self.schedule.add(agent)
        self.grid.place_agent(agent, graph_id)

    def step(self):
        self.choose_agent()
        self.schedule.step()
        # Save the statistics
        self.data_collector.collect(self)

    def choose_agent(self):
        # weighted random choice based on agents attentions
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
