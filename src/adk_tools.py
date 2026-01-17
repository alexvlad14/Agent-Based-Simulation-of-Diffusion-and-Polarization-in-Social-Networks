import networkx as nx
from src.data_loader import load_graph
from src.tools_graph import (
    init_store, 
    get_neighbors, 
    shortest_path, 
    compute_centrality, 
    detect_communities
)
from src.diffusion import (
    SimulationConfig, 
    experiment_influencers_vs_random, 
    experiment_polarized_communities
)

# --- GLOBAL STATE ---
# Εδώ κρατάμε τον γράφο και τα αποτελέσματα στη μνήμη
STATE = {
    "G": None,          # Το αντικείμενο NetworkX Graph
    "last_result": None # Τα αποτελέσματα της τελευταίας προσομοίωσης
}

# --- HELPER ---
def _get_store_factory():
    """Returns a function that creates a fresh GraphStore from the loaded graph."""
    if STATE["G"] is None:
        raise ValueError("Graph is not loaded. Please call load_graph_facebook first.")
    # Επιστρέφουμε lambda που φτιάχνει νέο store κάθε φορά (όπως θέλει το diffusion.py)
    return lambda: init_store(STATE["G"])

# --- AGENT TOOLS ---

def load_graph_facebook():
    """
    Loads the Facebook graph data from ./data/facebook_combined.txt into memory.
    Useful for initializing the system.
    """
    path = "./data/facebook_combined.txt"
    try:
        G = load_graph(path)
        STATE["G"] = G
        return f"Graph loaded successfully. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
    except Exception as e:
        return f"Error loading graph: {str(e)}"

def tool_neighbors(node_id: str):
    """
    Returns the neighbors of a specific node in the graph.
    Args:
        node_id: The ID of the node (e.g., '0', '100').
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        # Φτιάχνουμε προσωρινό store για να καλέσουμε τη συνάρτησή σου
        store = init_store(STATE["G"])
        nid = int(node_id)
        neighs = get_neighbors(store, nid)
        return f"Node {nid} has {len(neighs)} neighbors. First 10: {neighs[:10]}"
    except Exception as e:
        return f"Error: {e}"

def tool_shortest_path(source: str, target: str):
    """
    Calculates the shortest path between two nodes.
    Args:
        source: Start node ID.
        target: End node ID.
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        store = init_store(STATE["G"])
        res = shortest_path(store, int(source), int(target))
        if res["length"] == float('inf'):
            return "No path found."
        return f"Path found (Length {res['length']}): {res['path']}"
    except Exception as e:
        return f"Error: {e}"

def tool_centrality():
    """
    Calculates and returns the top 10 nodes by Degree Centrality.
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        store = init_store(STATE["G"])
        res = compute_centrality(store, metric="degree")
        # Μορφοποίηση για να είναι ευανάγνωστο από το LLM
        top_str = ", ".join([f"{n}: {s:.4f}" for n, s in res["top10"]])
        return f"Top 10 Nodes by Degree Centrality: {top_str}"
    except Exception as e:
        return f"Error: {e}"

def tool_communities():
    """
    Detects communities using Louvain algorithm and returns metadata.
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        store = init_store(STATE["G"])
        res = detect_communities(store)
        return f"Community Detection (Louvain): Found {res['num_communities']} communities."
    except Exception as e:
        return f"Error: {e}"

def run_influencers_vs_random(k: int = 10, T: int = 10):
    """
    Runs a diffusion experiment comparing Influencers vs Random seeds.
    Args:
        k: Number of seed nodes (default 10).
        T: Number of timesteps (default 10).
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        print(f"DEBUG: Running Influencers vs Random (k={k}, T={T})...")
        
        # 1. Ρύθμιση Config
        cfg = SimulationConfig(T=int(T), clamp=True)
        
        # 2. Εκτέλεση του πειράματος χρησιμοποιώντας το δικό σου diffusion.py
        factory = _get_store_factory()
        results = experiment_influencers_vs_random(factory, cfg, k=int(k))
        
        # 3. Αποθήκευση αποτελεσμάτων
        STATE["last_result"] = results
        
        # 4. Επιστροφή σύντομης περίληψης
        inf_reach = results['influencers']['final']['reach']
        rnd_reach = results['random']['final']['reach']
        return f"Experiment Completed. Influencer Reach: {inf_reach:.2f}, Random Reach: {rnd_reach:.2f}. Call summarize_last_result for details."
    except Exception as e:
        return f"Simulation Error: {e}"

def run_polarized(frac: float = 0.05):
    """
    Runs a diffusion experiment with polarized communities.
    Args:
        frac: Fraction of nodes to infect in each community (0.0 to 1.0).
    """
    if STATE["G"] is None: return "Graph not loaded."
    try:
        print(f"DEBUG: Running Polarized (frac={frac})...")
        
        cfg = SimulationConfig(T=30, clamp=True) # Default T=30
        factory = _get_store_factory()
        
        # Εκτέλεση δικού σου κώδικα
        result = experiment_polarized_communities(factory, cfg, frac=float(frac))
        
        STATE["last_result"] = result
        return f"Polarized Experiment Completed. Final Reach: {result['final']['reach']:.2f}. Call summarize_last_result for details."
    except Exception as e:
        return f"Simulation Error: {e}"

def summarize_last_result():
    """
    Analyzes and explains the results of the last simulation run.
    """
    res = STATE["last_result"]
    if res is None:
        return "No simulation has been run yet."
    
    # Έλεγχος: Είναι Influencers vs Random ή Polarized;
    try:
        if "influencers" in res and "random" in res:
            # Είναι Influencers vs Random
            inf = res["influencers"]
            rnd = res["random"]
            return (
                f"--- Comparison Results ---\n"
                f"Influencer Strategy:\n"
                f"  - Final Reach: {inf['final']['reach']:.2%}\n"
                f"  - Time to 30%: {inf['time_to_30pct']} steps\n"
                f"  - Active Nodes: {inf['final']['active_nodes_count']}\n\n"
                f"Random Strategy:\n"
                f"  - Final Reach: {rnd['final']['reach']:.2%}\n"
                f"  - Time to 30%: {rnd['time_to_30pct']} steps"
            )
        elif "final" in res:
            # Είναι Polarized (μονό αποτέλεσμα)
            fin = res["final"]
            return (
                f"--- Polarized Diffusion Results ---\n"
                f"Final Reach: {fin['reach']:.2%}\n"
                f"Entropy: {fin['entropy']:.4f}\n"
                f"Bimodality: {fin['bimodality']:.4f} (High means polarized)\n"
                f"Active Nodes: {fin['active_nodes_count']}"
            )
        else:
            return f"Unknown result format: {res.keys()}"
    except Exception as e:
        return f"Error parsing results: {e}"