from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import networkx as nx
from src.tools_graph import GraphStore, init_store

@dataclass
class AppState:
    G: Optional[nx.Graph] = None
    store: Optional[GraphStore] = None
    last_result: Optional[Dict[str, Any]] = None

STATE = AppState()

def ensure_store(G: nx.Graph) -> GraphStore:
    if STATE.store is None:
        STATE.G = G
        STATE.store = init_store(G)
    return STATE.store

