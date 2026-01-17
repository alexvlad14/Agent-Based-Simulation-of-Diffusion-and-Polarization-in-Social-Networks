from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import math

import numpy as np


def _vals(opinions: Dict[int, float]) -> np.ndarray:
    return np.array(list(opinions.values()), dtype=float)


def diffusion_reach(opinions: Dict[int, float], thr: float = 0.7) -> float:
    """
    Fraction of nodes whose absolute opinion magnitude exceeds thr.
    """
    v = _vals(opinions)
    if v.size == 0:
        return 0.0
    return float(np.mean(np.abs(v) >= thr))


def time_to_reach(reach_series: List[float], target: float = 0.5) -> Optional[int]:
    """
    First timestep where reach >= target. Returns None if never reached.
    """
    for t, r in enumerate(reach_series):
        if r >= target:
            return t
    return None


def opinion_mean(opinions: Dict[int, float]) -> float:
    v = _vals(opinions)
    if v.size == 0:
        return 0.0
    return float(np.mean(v))


def opinion_variance(opinions: Dict[int, float]) -> float:
    """
    Variance as a simple polarization proxy.
    """
    v = _vals(opinions)
    if v.size == 0:
        return 0.0
    return float(np.var(v))


def opinion_bimodality_proxy(opinions: Dict[int, float], bins: int = 21) -> float:
    """
    Simple bimodality proxy:
    Measures how "two-peaked" the distribution is by comparing mass near extremes vs center.

    Returns value in [0,1] approximately; higher indicates more mass at extremes.
    """
    v = _vals(opinions)
    if v.size == 0:
        return 0.0

    # Histogram on [-1, 1]
    hist, edges = np.histogram(v, bins=bins, range=(-1.0, 1.0), density=False)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0

    # Define center region ~ around 0 and extreme regions near -1 and +1
    # Center bins: middle 1/3, extremes: outer 1/3
    third = max(1, bins // 3)
    left_ext = hist[:third].sum()
    right_ext = hist[-third:].sum()
    center = hist[third:-third].sum() if bins > 2 * third else 0.0

    # More extremes, less center -> higher
    score = (left_ext + right_ext) / total - (center / total)
    # shift/scale to roughly [0,1]
    score = (score + 1.0) / 2.0
    return float(max(0.0, min(1.0, score)))


def opinion_entropy(opinions: Dict[int, float], bins: int = 21) -> float:
    """
    Shannon entropy over histogram bins on [-1,1].
    Higher entropy => more spread / uncertainty.
    """
    v = _vals(opinions)
    if v.size == 0:
        return 0.0

    hist, _ = np.histogram(v, bins=bins, range=(-1.0, 1.0), density=False)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0

    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def inter_community_spread(
    opinions: Dict[int, float],
    communities: Dict[int, int],
    thr: float = 0.7
) -> Dict[str, Any]:
    """
    Measures how spread differs across communities.

    Returns:
      - community_reach: dict[community_id] -> reach within that community
      - mean_reach: average community reach
      - std_reach: dispersion across communities
      - active_communities: how many communities have at least 1 node above threshold
    """
    # group nodes by community
    comm_to_nodes: Dict[int, List[int]] = {}
    for n in opinions.keys():
        if n not in communities:
            continue
        cid = int(communities[n])
        comm_to_nodes.setdefault(cid, []).append(n)

    community_reach: Dict[int, float] = {}
    active = 0
    for cid, nodes in comm_to_nodes.items():
        vals = np.array([opinions[n] for n in nodes], dtype=float)
        r = float(np.mean(np.abs(vals) >= thr)) if vals.size else 0.0
        community_reach[cid] = r
        if np.any(np.abs(vals) >= thr):
            active += 1

    reaches = np.array(list(community_reach.values()), dtype=float)
    if reaches.size == 0:
        return {
            "community_reach": {},
            "mean_reach": 0.0,
            "std_reach": 0.0,
            "active_communities": 0
        }

    return {
        "community_reach": community_reach,
        "mean_reach": float(np.mean(reaches)),
        "std_reach": float(np.std(reaches)),
        "active_communities": int(active)
    }


def centrality_overlap(
    changed_nodes: List[int],
    centrality_scores: Dict[int, float],
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Measures overlap between nodes that changed (or became active) and top-k central nodes.

    changed_nodes: list of node ids you deem "impactful" (e.g., nodes with |opinion|>=thr at final step)
    centrality_scores: dict node->score
    """
    top_nodes = [n for n, _ in sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    changed_set = set(changed_nodes)
    top_set = set(top_nodes)

    inter = changed_set & top_set
    precision = len(inter) / len(top_set) if top_set else 0.0
    recall = len(inter) / len(changed_set) if changed_set else 0.0

    return {
        "top_k": int(top_k),
        "changed_count": int(len(changed_set)),
        "overlap_count": int(len(inter)),
        "precision": float(precision),
        "recall": float(recall),
        "overlap_nodes_sample": list(sorted(inter))[:20],
    }


def summarize_step(
    opinions: Dict[int, float],
    thr: float = 0.7,
    bins: int = 21
) -> Dict[str, Any]:
    """
    Convenience function to compute a compact set of metrics at a timestep.
    """
    return {
        "reach": diffusion_reach(opinions, thr=thr),
        "mean": opinion_mean(opinions),
        "variance": opinion_variance(opinions),
        "bimodality": opinion_bimodality_proxy(opinions, bins=bins),
        "entropy": opinion_entropy(opinions, bins=bins),
    }


def active_nodes(opinions: Dict[int, float], thr: float = 0.7) -> List[int]:
    """
    Nodes considered 'active' at threshold thr.
    """
    return [int(n) for n, v in opinions.items() if abs(float(v)) >= thr]
