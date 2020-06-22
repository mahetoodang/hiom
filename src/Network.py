import networkx as nx
import numpy as np


class Network:
    """
    A class used to create a representation of population and connections between individuals.

    Attributes
    ----------
    params : dict
        Dictionary containing parameters used to generate network.
        Fields:
            "method" : string
                Network will be generated according to selected method. Possible values: "er", "ba", 
                "ws", "sb", "lattice" and "social media". Each method has a specific set of parameters 
                which enable more precise control of their results via other fields in this dictionary. 
                Examples are provided in docstings of the methods. 
            p, k, m : floats
                Parameters used to cofigure the network generation method more precisely. Their precise 
                meaning depends on the selected method, therefore for more information consult docstrings of 
                a function responsible for a particular method.
            path : string
                Path to the file containing graph's list of edges used to create social media graph.
            n_blocks : int
                Number of components to be created in the stochastic block method
    n : int
        Desired number of nodes in the network

    Methods
    -------
    get_graph : Graph
        Returns a graph created according to the parameters
    """
    def __init__(self, params, n=100):
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
        Example parameter dictionary: { "method": "er", "p": 0.5 } 

        self.n : int
            Total number of nodes in a graph
        self.p : float
            Probability of creating edge between two nodes
        """

        self.G = nx.fast_gnp_random_graph(self.n, self.params['p'])

    def create_ba_graph(self):
        """
        Generating a Barabasi-Albert network.
        Example parameter dictionary: { "method": "ba", "m": 2 } 

        self.n : int
            Number of nodes
        self.m : int
            Number of edges to attach from a new node to existing ones
        """

        self.G = nx.barabasi_albert_graph(self.n, self.params['m'])

    def create_ws_graph(self):
        """
        Generating a Watts-Stogatz network.
        Example parameter dictionary: { "method": "ws", "p": 0.2, "k": 2 } 

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
        Generating a stochastic block graph.
        Example parameter dictionary: { "method": "sb", "n_blocks": 10, "p": 0.1, "k": 0.01 } 

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
        Generating a 2d lattice grid.
        Example parameter dictionary: { "method": "lattice", "m": 100 } 

        self.m : int
            Size of 1st dimension
        self.n : int
            Size of 2nd dimension
        """
        self.G = nx.grid_2d_graph(self.params['m'], self.n)

    def create_social_media_graph(self):
        """
        Generating a network from a data file containing a list of edges.
        Example parameter dictionary: { "method": "social_media", "path": "../data/facebook_combined.txt" } 

        self.path : string
            Path to the list of edges file
        """
        self.G = nx.read_edgelist(self.params['path'])
