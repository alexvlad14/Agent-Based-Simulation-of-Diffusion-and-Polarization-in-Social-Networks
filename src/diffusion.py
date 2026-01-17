from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import random

import numpy as np

from src.tools_graph import GraphStore, set_opinion, get_opinion_vector, apply_degroot_step, compute_centrality, detect_communities
from src.metrics import summarize_step, active_nodes, inter_community_spread, time_to_reach, centrality_overlap


@dataclass
class SimulationConfig:
    T: int = 30
    alpha: float = 0.65
    thr: float = 0.7
    seed: int = 42
    init_mu: float = 0.0
    init_sigma: float = 0.15
    clamp: bool = True


def init_random_opinions(store: GraphStore, cfg: SimulationConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    for n in store.G.nodes():
        v = float(rng.normal(cfg.init_mu, cfg.init_sigma))
        v = max(-1.0, min(1.0, v))
        store._opinions[int(n)] = v


def seed_influencers(
    store: GraphStore,
    k: int = 10,
    metric: str = "degree",
    value: float = 1.0
) -> Dict[str, Any]:
    """
    Seed top-k nodes by a centrality metric with a fixed opinion value.
    """
    cent = compute_centrality(store, metric)
    topk = [n for n, _ in cent["top10"][:k]] if k <= 10 else \
        [n for n, _ in sorted(cent["scores"].items(), key=lambda x: x[1], reverse=True)[:k]]

    for n in topk:
        set_opinion(store, int(n), float(value))

    return {"strategy": "influencers", "metric": metric, "k": int(k), "seed_nodes": [int(x) for x in topk]}


def seed_random_nodes(
    store: GraphStore,
    k: int = 10,
    value: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Seed k random nodes with a fixed opinion value.
    """
    rng = random.Random(seed)
    nodes = list(store.G.nodes())
    chosen = rng.sample(nodes, k)

    for n in chosen:
        set_opinion(store, int(n), float(value))

    return {"strategy": "random", "k": int(k), "seed_nodes": [int(x) for x in chosen]}


def seed_polarized_communities(
    store: GraphStore,
    community_a: int,
    community_b: int,
    frac: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Polarized seeding:
    - select a fraction of nodes from community_a set to +1
    - select a fraction of nodes from community_b set to -1
    """
    comm = detect_communities(store)
    mapping = comm["communities"]

    a_nodes = [n for n, cid in mapping.items() if cid == community_a]
    b_nodes = [n for n, cid in mapping.items() if cid == community_b]

    if len(a_nodes) == 0 or len(b_nodes) == 0:
        raise ValueError("Selected communities have no nodes. Choose different community ids.")

    rng = random.Random(seed)
    ka = max(1, int(frac * len(a_nodes)))
    kb = max(1, int(frac * len(b_nodes)))

    a_seed = rng.sample(a_nodes, ka)
    b_seed = rng.sample(b_nodes, kb)

    for n in a_seed:
        set_opinion(store, int(n), 1.0)
    for n in b_seed:
        set_opinion(store, int(n), -1.0)

    return {
        "strategy": "polarized_communities",
        "community_a": int(community_a),
        "community_b": int(community_b),
        "frac": float(frac),
        "seed_nodes_pos": [int(x) for x in a_seed],
        "seed_nodes_neg": [int(x) for x in b_seed],
        "community_method": comm["method"],
        "num_communities": comm["num_communities"],
    }



def run_simulation(
    store: GraphStore,
    cfg: SimulationConfig,
    seeding_info: Dict[str, Any],
    compute_intercommunity: bool = True,
    centrality_metric_for_overlap: str = "degree",
    top_k_overlap: int = 200
) -> Dict[str, Any]:
    """
    Run diffusion for T steps and collect metrics time series.
    """
    series: List[Dict[str, Any]] = []

    # Pre-compute communities mapping if needed
    comm_mapping = None
    comm_meta = None
    if compute_intercommunity:
        comm = detect_communities(store)
        comm_mapping = comm["communities"]
        comm_meta = {k: comm[k] for k in ["method", "num_communities", "cached"] if k in comm}

    # Pre-compute centrality scores for overlap evaluation
    cent_payload = compute_centrality(store, centrality_metric_for_overlap)
    cent_scores = cent_payload["scores"]

    # timestep 0 metrics
    opinions0 = get_opinion_vector(store)
    snap0 = summarize_step(opinions0, thr=cfg.thr)
    if compute_intercommunity and comm_mapping is not None:
        snap0["intercommunity"] = inter_community_spread(opinions0, comm_mapping, thr=cfg.thr)
    series.append(snap0)
    
    stubborn = set()
    if "seed_nodes" in seeding_info:
        stubborn = set(seeding_info["seed_nodes"])
    elif "seed_nodes_pos" in seeding_info and "seed_nodes_neg" in seeding_info:
        stubborn = set(seeding_info["seed_nodes_pos"]) | set(seeding_info["seed_nodes_neg"])




    # diffusion loop
    for t in range(1, cfg.T + 1):
        apply_degroot_step(store, alpha=cfg.alpha, clamp=cfg.clamp, stubborn_nodes=stubborn)
        opinions_t = get_opinion_vector(store)
        snap = summarize_step(opinions_t, thr=cfg.thr)

        if compute_intercommunity and comm_mapping is not None:
            snap["intercommunity"] = inter_community_spread(opinions_t, comm_mapping, thr=cfg.thr)

        series.append(snap)

    # summary at end
    reach_series = [s["reach"] for s in series]
    t50 = time_to_reach(reach_series, target=0.50)
    t30 = time_to_reach(reach_series, target=0.30)

    final_op = get_opinion_vector(store)
    final_active = active_nodes(final_op, thr=cfg.thr)

    overlap = centrality_overlap(final_active, cent_scores, top_k=top_k_overlap)

    return {
        "config": cfg.__dict__,
        "seeding": seeding_info,
        "series": series,
        "time_to_30pct": t30,
        "time_to_50pct": t50,
        "final": {
            "reach": series[-1]["reach"],
            "variance": series[-1]["variance"],
            "bimodality": series[-1]["bimodality"],
            "entropy": series[-1]["entropy"],
            "active_nodes_count": len(final_active),
        },
        "overlap_with_centrality": {
            "metric": centrality_metric_for_overlap,
            **overlap
        },
        "communities_meta": comm_meta
    }


def experiment_influencers_vs_random(
    store_factory,
    cfg: SimulationConfig,
    k: int = 50,
    metric: str = "degree"
) -> Dict[str, Any]:
    """
    Runs two experiments on fresh stores:
      A) influencer seeding (top-k by metric)
      B) random seeding (k)
    store_factory: callable that returns a fresh GraphStore (same graph) each time.
    """
    # A) influencers
    store_a = store_factory()
    init_random_opinions(store_a, cfg)
    seed_a = seed_influencers(store_a, k=k, metric=metric, value=1.0)
    res_a = run_simulation(store_a, cfg, seed_a, compute_intercommunity=True)

    # B) random
    store_b = store_factory()
    init_random_opinions(store_b, cfg)
    seed_b = seed_random_nodes(store_b, k=k, value=1.0, seed=cfg.seed)
    res_b = run_simulation(store_b, cfg, seed_b, compute_intercommunity=True)

    return {"influencers": res_a, "random": res_b}


def experiment_polarized_communities(
    store_factory,
    cfg: SimulationConfig,
    frac: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs polarized community diffusion on a fresh store.
    Chooses two largest communities automatically to maximize signal.
    """
    store = store_factory()
    init_random_opinions(store, cfg)

    comm = detect_communities(store)
    mapping = comm["communities"]

    # find largest communities
    counts: Dict[int, int] = {}
    for _, cid in mapping.items():
        counts[int(cid)] = counts.get(int(cid), 0) + 1

    largest = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
    if len(largest) < 2:
        raise ValueError("Not enough communities found for polarized experiment.")

    cA, _ = largest[0]
    cB, _ = largest[1]

    seed_info = seed_polarized_communities(store, cA, cB, frac=frac, seed=seed)
    res = run_simulation(store, cfg, seed_info, compute_intercommunity=True)
    return res

