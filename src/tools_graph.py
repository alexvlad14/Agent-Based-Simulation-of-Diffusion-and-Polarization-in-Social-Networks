from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import math

import networkx as nx

try:
    # python-louvain package
    import community as community_louvain  # type: ignore
except Exception:
    community_louvain = None


CentralityName = Union[
    str  # "degree", "betweenness", "pagerank", "eigenvector"
]


@dataclass
class GraphStore:
    """
    In-memory graph store with cached computations.
    Designed to back tool-like functions for agents.

    Notes
    -----
    - Stores a single NetworkX graph.
    - Caches centralities and communities to avoid recomputation.
    """
    G: nx.Graph
    _centrality_cache: Dict[str, Dict[int, float]] = field(default_factory=dict)
    _communities_cache: Optional[Dict[int, int]] = None
    _shortest_path_cache: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    _opinions: Dict[int, float] = field(default_factory=dict)

    def has_node(self, node: int) -> bool:
        return self.G.has_node(node)

    def require_node(self, node: int) -> None:
        if not self.has_node(node):
            raise ValueError(f"Node id not found in graph: {node}")

    def set_opinion(self, node: int, value: float) -> None:
        self.require_node(node)
        # clamp to [-1, 1]
        value = max(-1.0, min(1.0, float(value)))
        self._opinions[node] = value

    def get_opinion(self, node: int, default: float = 0.0) -> float:
        self.require_node(node)
        return float(self._opinions.get(node, default))


# -----------------------------
# Tool-like functions
# -----------------------------

def init_store(G: nx.Graph) -> GraphStore:
    """
    Initialize a GraphStore for a given graph.
    """
    return GraphStore(G=G)


def get_neighbors(store: GraphStore, node_id: int) -> List[int]:
    """
    Return neighbors of a node.
    """
    store.require_node(node_id)
    return [int(n) for n in store.G.neighbors(node_id)]


def shortest_path(store: GraphStore, src: int, dst: int) -> Dict[str, Any]:
    """
    Return shortest path (unweighted) between src and dst as a list of node ids.
    Caches results.
    """
    store.require_node(src)
    store.require_node(dst)

    key = (int(src), int(dst))
    if key in store._shortest_path_cache:
        path = store._shortest_path_cache[key]
        return {"src": src, "dst": dst, "path": path, "length": len(path) - 1, "cached": True}

    try:
        path = nx.shortest_path(store.G, source=src, target=dst)
        path = [int(x) for x in path]
        store._shortest_path_cache[key] = path
        return {"src": src, "dst": dst, "path": path, "length": len(path) - 1, "cached": False}
    except nx.NetworkXNoPath:
        return {"src": src, "dst": dst, "path": [], "length": math.inf, "cached": False}


def compute_centrality(store: GraphStore, metric: CentralityName = "degree") -> Dict[str, Any]:
    """
    Compute and return a centrality dict {node: score}.
    Supported:
      - degree
      - betweenness
      - pagerank
      - eigenvector

    Returns a payload with metadata and sorted top-10.
    """
    metric = str(metric).strip().lower()
    if metric in store._centrality_cache:
        scores = store._centrality_cache[metric]
        top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return {"metric": metric, "scores": scores, "top10": top10, "cached": True}

    if metric == "degree":
        scores = nx.degree_centrality(store.G)
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(store.G, normalized=True)
    elif metric == "pagerank":
        scores = nx.pagerank(store.G)
    elif metric == "eigenvector":
        # may fail to converge for some graphs; provide fallback
        try:
            scores = nx.eigenvector_centrality(store.G, max_iter=2000, tol=1e-6)
        except Exception:
            scores = nx.eigenvector_centrality_numpy(store.G)
    else:
        raise ValueError(f"Unsupported centrality metric: {metric}")

    scores = {int(k): float(v) for k, v in scores.items()}
    store._centrality_cache[metric] = scores
    top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"metric": metric, "scores": scores, "top10": top10, "cached": False}


def detect_communities(store: GraphStore) -> Dict[str, Any]:
    """
    Detect communities using Louvain if available.
    Returns a mapping {node: community_id} and basic stats.
    """
    if store._communities_cache is not None:
        comms = store._communities_cache
        return {
            "method": "louvain",
            "communities": comms,
            "num_communities": len(set(comms.values())),
            "cached": True
        }

    if community_louvain is None:
        raise ImportError(
            "python-louvain not installed. Install with: pip install python-louvain"
        )

    partition = community_louvain.best_partition(store.G)  # {node: community_id}
    comms = {int(k): int(v) for k, v in partition.items()}
    store._communities_cache = comms

    return {
        "method": "louvain",
        "communities": comms,
        "num_communities": len(set(comms.values())),
        "cached": False
    }


def set_opinion(store: GraphStore, node_id: int, value: float) -> Dict[str, Any]:
    """
    Set a node opinion value in [-1, 1].
    """
    store.set_opinion(node_id, value)
    return {"node": int(node_id), "opinion": store.get_opinion(node_id), "status": "ok"}


def get_opinion(store: GraphStore, node_id: int, default: float = 0.0) -> Dict[str, Any]:
    """
    Get a node opinion value. If not set, returns default.
    """
    val = store.get_opinion(node_id, default=default)
    return {"node": int(node_id), "opinion": float(val)}


def get_opinion_vector(store: GraphStore, default: float = 0.0) -> Dict[int, float]:
    """
    Return opinions for all nodes as a dict.
    Nodes without explicit opinion get default.
    """
    out: Dict[int, float] = {}
    for n in store.G.nodes():
        out[int(n)] = float(store._opinions.get(int(n), default))
    return out


from typing import Optional, Set  # βάλε το αν δεν υπάρχει ήδη

def apply_degroot_step(
    store: GraphStore,
    alpha: float = 0.65,
    self_weight: float = 0.0,
    default: float = 0.0,
    clamp: bool = True,
    stubborn_nodes: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    Apply one DeGroot-like diffusion step with optional stubborn (zealot) nodes.
    Stubborn nodes keep their opinion fixed across steps.
    """
    alpha = float(alpha)
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0,1]")

    stubborn_nodes = stubborn_nodes or set()

    opinions = get_opinion_vector(store, default=default)
    new_op: Dict[int, float] = {}

    for n in store.G.nodes():
        n = int(n)

        # Zealots keep their value fixed
        if n in stubborn_nodes:
            new_op[n] = float(opinions[n])
            continue

        neigh = list(store.G.neighbors(n))
        if not neigh:
            v = opinions[n]
        else:
            neigh_avg = sum(opinions[int(u)] for u in neigh) / len(neigh)
            v = (1 - alpha) * opinions[n] + alpha * neigh_avg

            if self_weight != 0.0:
                v = (1 - self_weight) * v + self_weight * opinions[n]

        if clamp:
            v = max(-1.0, min(1.0, float(v)))
        new_op[n] = float(v)

    store._opinions = new_op
    return {"status": "ok", "alpha": alpha, "updated_nodes": len(new_op), "stubborn": len(stubborn_nodes)}

