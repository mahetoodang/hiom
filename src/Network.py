import networkx as nx


class Network:
    """
    A class used to create a representation of population and connections between individuals.

    Attributes
    method : str
        String identifying a model of the network: "er", "ba", "ws", "lattice" and "social media"
    Other attributes and their purpose depend on the selected method.

    Methods
    get_graph : Graph
        Returns a graph created according to the parameters
    """
    def __init__(self, method="er", n=100, p=0.5, m=100, k=2, path="../data/facebook_combined.txt"):
        self.G = nx.Graph()
        self.n = n
        self.p = p
        self.m = m
        self.k = k
        self.path = path
        self.init_methods = {"er": self.create_random_graph,
                             "ba": self.create_ba_graph,
                             "ws": self.create_ws_graph,
                             "lattice": self.create_lattice,
                             "social_media": self.create_social_media_graph}
        self.init_methods[method]()

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

        self.G = nx.fast_gnp_random_graph(self.n, self.p)

    def create_ba_graph(self):
        """
        Generating a Barabasi-Albert network

        self.n : int
            Number of nodes
        self.m : int
            Number of edges to attach from a new node to existing ones
        """

        self.G = nx.barabasi_albert_graph(self.n, self.m)

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

        self.G = nx.watts_strogatz_graph(self.n, self.k, self.p)

    def create_lattice(self):
        """
        Generating a 2d lattice grid

        self.m : int
            Size of 1st dimension
        self.n : int
            Size of 2nd dimension
        """
        self.G = nx.grid_2d_graph(self.m, self.n)

    def create_social_media_graph(self):
        """
        Generating a network from a data file containing a list of edges.

        self.path : string
            Path to the list of edges file
        """
        self.G = nx.read_edgelist(self.path)
