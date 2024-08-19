import networkx as nx
import numpy as np
import abc

class FeatureGen(metaclass=abc.ABCMeta):
    """Feature Generator base class."""
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

    @abc.abstractmethod
    def gen_edge_features(self, G):
        pass


class ConstFeatureGen(FeatureGen):
    """Constant Feature class."""
    def __init__(self, val, max_degree=20):
        self.val = val
        self.max_degree = max_degree
        self.one_hots = np.eye(max_degree, dtype=np.float32)

    def gen_node_features(self, G):
        feat_dict = {}
        if self.val is None:
            for k, v in G.degree():
                if v < self.max_degree:
                    feat_dict[k] = {'feat': self.one_hots[v]}
                else:
                    feat_dict[k] = {'feat': self.one_hots[self.max_degree - 1]}  
        else:
            feat_dict = {i:{'feat': np.array(self.val, dtype=np.float32)} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

    def gen_edge_features(self, G):
        # Example: All edges get a constant feature (1.0)
        feat_dict = {edge: {'feat': np.array([1.0], dtype=np.float32)} for edge in G.edges()}
        nx.set_edge_attributes(G, feat_dict)


class GaussianFeatureGen(FeatureGen):
    """Gaussian Feature class."""
    def __init__(self, mu, sigma):
        self.mu = mu
        if sigma.ndim < 2:
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {i: {"feat": feat[i]} for i in range(feat.shape[0])}
        nx.set_node_attributes(G, feat_dict)

    def gen_edge_features(self, G):
        # Example: Use absolute difference of node features as edge feature
        feat_dict = {}
        for u, v in G.edges():
            diff = np.abs(G.nodes[u]['feat'] - G.nodes[v]['feat'])
            feat_dict[(u, v)] = {'feat': diff}
        nx.set_edge_attributes(G, feat_dict)


class GridFeatureGen(FeatureGen):
    """Grid Feature class."""
    def __init__(self, mu, sigma, com_choices):
        self.mu = mu                    # Mean
        self.sigma = sigma              # Variance
        self.com_choices = com_choices  # List of possible community labels

    def gen_node_features(self, G):
        # Generate community assignment
        community_dict = {
            n: self.com_choices[0] if G.degree(n) < 4 else self.com_choices[1]
            for n in G.nodes()
        }

        # Generate random variable
        s = np.random.normal(self.mu, self.sigma, G.number_of_nodes())

        # Generate features
        feat_dict = {
            n: {"feat": np.asarray([community_dict[n], s[i]])}
            for i, n in enumerate(G.nodes())
        }

        nx.set_node_attributes(G, feat_dict)
        return community_dict

    def gen_edge_features(self, G):
        # Example: Generate binary edge feature based on community labels
        feat_dict = {}
        for u, v in G.edges():
            same_community = int(G.nodes[u]['feat'][0] == G.nodes[v]['feat'][0])
            feat_dict[(u, v)] = {'feat': np.array([same_community], dtype=np.float32)}
        nx.set_edge_attributes(G, feat_dict)
