import networkx as nx
import os

def load_graph(file_path):
    """
    Loads a graph from a text file containing an edge list.
    
    Args:
        file_path (str): The path to the edge list file.
        
    Returns:
        networkx.Graph: The loaded graph.
    """
    print(f"Attempting to load graph from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
        
    # Το facebook_combined.txt είναι συνήθως edge list με κενά
    G = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
    
    return G