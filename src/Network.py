import networkx as nx
import numpy as np


class Network:
    """
    A class used to create a representation of population and connections between individuals.

    Attributes
    params : dict
        Dictionary containing parameters used to generate network.
        Fields:
            "method" : string
                Network will be generated according to selected method. Possible choices: "er", "ba", "ws", "lattice" and "social media"
    n : int
        Desired number of nodes in the network

    Methods
    get_graph : Graph
        Returns a graph created according to the parameters
    """
    def __init__(self, params, n=100): #p=0.5, m=100, k=2, path="../data/facebook_combined.txt", n_blocks=10):
        self.G = nx.Graph()
        self.n = n
        self.params = params
        self.init_methods = {"er": self.create_random_graph,
                             "ba": self.create_ba_graph,
                             "ws": self.create_ws_graph,
                             "sb": self.create_sb_graph,
                             "lattice": self.create_lattice,
                             "social_media": self.create_social_media_graph}
        self.init_methods[params['method']]()

    def get_graph(self):
        return self.G

    def create_random_graph(self):
        """
        Generating a random (Erdos-Renyi) network.

        self.n : int
            Total number of nodes in a graph
        self.p : float
            Probability of creating edge between two nodes
        """

        self.G = nx.fast_gnp_random_graph(self.n, self.params['p'])

    def create_ba_graph(self):
        """
        Generating a Barabasi-Albert network

        self.n : int
            Number of nodes
        self.m : int
            Number of edges to attach from a new node to existing ones
        """

        self.G = nx.barabasi_albert_graph(self.n, self.params['m'])

    def create_ws_graph(self):
        """
        Generating a Watts-Stogatz network

        self.n : int
            Number of nodes
        self.k : int
            Number of neighbors to which a node is connected initially in the ring topology
        self.p : float
            Probability of rewiring: adding a new node and removing existing one
        """

        self.G = nx.watts_strogatz_graph(self.n, self.params['k'], self.params['p'])

    def create_sb_graph(self):
        """
        Generating a stochastic block graph

        self.p : float
            Probability of the edge between nodes belonging to different components
        self.k : float
            Probability of edges within a block/community
        self.n_blocks : int
            Number of blocks/communities that are to be created within the network
        """
        n_blocks = self.params['n_blocks']
        probabilities = np.full((n_blocks, n_blocks), self.params['p'])
        np.fill_diagonal(probabilities, self.params['k'])
        block_sizes = [int(self.n/n_blocks) for _ in range(n_blocks)]
        self.G = nx.stochastic_block_model(block_sizes, probabilities)

    def create_lattice(self):
        """
        Generating a 2d lattice grid

        self.m : int
            Size of 1st dimension
        self.n : int
            Size of 2nd dimension
        """
        self.G = nx.grid_2d_graph(self.params['m'], self.n)

    def create_social_media_graph(self):
        """
        Generating a network from a data file containing a list of edges.

        self.path : string
            Path to the list of edges file
        """
        self.G = nx.read_edgelist(self.params['path'])
