import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors
plt.switch_backend("agg")
import networkx as nx
import numpy as np
import synthetic_structsim
import featgen
import pdb

def perturb(graph_list, p):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list

def generate_graph(basis_type="ba", 
                   shape="house", 
                   nb_shapes=80, 
                   width_basis=300, 
                   feature_generator=None, 
                   m=5, 
                   random_edges=0.0):
    """ Generate a synthetic graph with specified base and shape substructures.

    Args:
        basis_type (str): Type of base graph (e.g., 'ba' for Barabasi-Albert).
        shape (str): The shape to attach to the base graph ('house', 'cycle', 'diamond', 'grid').
        nb_shapes (int): Number of shapes to attach.
        width_basis (int): Width of the base graph.
        feature_generator (FeatureGenerator): Generator for node features.
        m (int): Number of edges to attach from a new node to existing nodes (for BA graph).
        random_edges (float): Proportion of edges to randomly add to the graph.

    Returns:
        G (networkx.Graph): The generated graph.
        role_id (list): List of role IDs for each node.
    """
    if shape == "house":
        list_shapes = [["house"]] * nb_shapes
    elif shape == "cycle":
        list_shapes = [["cycle", 6]] * nb_shapes
    elif shape == "diamond":
        list_shapes = [["diamond"]] * nb_shapes
    elif shape == "grid":
        list_shapes = [["grid"]] * nb_shapes
    else:
        assert False, f"Unknown shape: {shape}"
    
    G, role_id, _ = synthetic_structsim.build_graph(width_basis, 
                                                    basis_type, 
                                                    list_shapes, 
                                                    rdm_basis_plugins=True, 
                                                    start=0, 
                                                    m=m)
                                            
    if random_edges != 0:
        G = perturb([G], random_edges)[0]
    
    if feature_generator is not None:
        feature_generator.gen_node_features(G)
        
    return G, role_id
