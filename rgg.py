# requires networkx
# requires numpy
# requires matplotlib

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class RandomGeometricGraph:

    """
    This class is used to create a spatial network.
    """

    def __init__(self, nodes, connection_type="hard", r_0=0.1, domain_type="rectangle", domain_height=1, domain_width=1, domain_radius=1) -> None:
        """
        Initializes a SpatialNetwork object.

        Parameters:
        nodes (list): A list of node coordinates to be included in the network.
        connection_type (str, optional): A string that names the function which decides whether two nodes are connected. Defaults to a "hard".
        r_0 (float, optional): The default distance threshold for the connection function. Defaults to 0.1.
        domain_type (str, optional): The type of domain for the network. Can be "rectangle" or "circle". Defaults to "rectangle". A rectangle has its bottom left corner on (0,0), and the circle is centred on (0,0).
        domain_height (int, optional): The height of the domain if it's a rectangle. Defaults to 1.
        domain_width (int, optional): The width of the domain if it's a rectangle. Defaults to 1.
        domain_radius (int, optional): The radius of the domain if it's a circle. Defaults to 1.
        """

        self.nodes = nodes
        self.connection_type = "hard"
        if self.connection_type == "hard":
            self.connection_function = lambda r: r <= r_0
        else:
            raise(ValueError, "Connection type not recognised, perhaps not implemented yet")
        self.r_0 = r_0
        self.domain_type = domain_type
        self.domain_height = domain_height
        self.domain_width = domain_width
        self.domain_radius = domain_radius

        if self.domain_type == "rectangle":
            for node in self.nodes:
                if node[0] > self.domain_width or node[1] > self.domain_height:
                    raise(ValueError, "Node coordinates are outside the specified domain")
        elif self.domain_type == "circle":
            for node in self.nodes:
                if node[0]**2 + node[1]**2 > self.domain_radius**2:
                    raise(ValueError, "Node coordinates are outside the specified domain")
        else:
            raise(ValueError, "Domain type not recognised, perhaps not implemented yet")
        
        self.edges = self.make_edges()    
        self.G = nx.Graph(self.edges)

    
    def make_edges(self) -> list:
        """
        Create edges from the list of nodes based on whether the connection function evaluates to True or False
        """

        edges = []
        for i in range(len(self.nodes)):
            for j in range(i, len(self.nodes)):
                if self.connection_type == "hard":
                    edge_ij = self.connection_function(r=np.linalg.norm(np.array(self.nodes[i]) - np.array(self.nodes[j])))
                    if edge_ij:
                        edges.append((i, j))

        return edges

    def plot(self) -> None:
        """
        Plot the network using matplotlib
        """

        plt.figure()
        plt.scatter(*zip(*self.nodes))

        # plot the edges
        for edge in self.edges:
            plt.plot([self.nodes[edge[0]][0], self.nodes[edge[1]][0]], [self.nodes[edge[0]][1], self.nodes[edge[1]][1]], 'k-')
        
        if self.domain_type == "circle":
            # plot a circle of radius domain radius
            theta = np.linspace(0, 2*np.pi, 100)
            x = self.domain_radius*np.cos(theta)
            y = self.domain_radius*np.sin(theta)
            plt.plot(x, y, 'k-')
            # make the axis square
            plt.axis('square')
        plt.show()

    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Returns the adjacency matrix of the network
        """

        return nx.adjacency_matrix(self.G)
    

    def get_laplacian_matrix(self) -> np.ndarray:
        """
        Returns the Laplacian matrix of the network
        """

        return nx.laplacian_matrix(self.G)
    

    def get_degree_matrix(self) -> np.ndarray:
        """
        Returns the degree matrix of the network
        """

        return nx.degree_matrix(self.G)
    

    def eigenvalues(self, normalised=True) -> np.ndarray:
        """
        Returns the eigenvalues of the Laplacian matrix of the network
        WARNING: This will convert the Laplacian to Dense format, which is not efficient for large networks
        """

        if normalised:
            L = self.get_laplacian_matrix()
            print(L)
            tr = L.diagonal().sum()
            return nx.laplacian_spectrum(self.G)/tr
        else:
            return nx.laplacian_spectrum(self.G)
        

    def plot_spectral_density(self, normalised=True) -> None:
        """
        Plots the spectral density of the network
        """

        plt.figure()
        plt.hist(self.eigenvalues(normalised=normalised))
        plt.show()