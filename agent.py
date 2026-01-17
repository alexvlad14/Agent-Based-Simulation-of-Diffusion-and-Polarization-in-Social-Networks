import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Εισαγωγή των δικών σου εργαλείων από το adk_tools.py
# (Αυτό το αρχείο το έχεις, οπότε το κρατάμε!)
from src.adk_tools import (
    load_graph_facebook,
    tool_neighbors,
    tool_shortest_path,
    tool_centrality,
    tool_communities,
    run_influencers_vs_random,
    run_polarized,
    summarize_last_result
)

#load_dotenv()

llm = ChatOpenAI(
    model="llama3.1:8b",  
    api_key="sk-dummy",
    base_url="https://babili.csd.auth.gr:11435/v1",
    temperature=0
)

# Λίστα εργαλείων
tools = [
    load_graph_facebook,
    tool_neighbors,
    tool_shortest_path,
    tool_centrality,
    tool_communities,
    run_influencers_vs_random,
    run_polarized,
    summarize_last_result
]

# Οδηγίες (System Prompt)
system_prompt = """
You are an SNA Agent. Your goal is to analyze the Facebook graph and run diffusion simulations.
1. ALWAYS load the graph first if not loaded.
2. Use 'run_influencers_vs_random' or 'run_polarized' for simulations.
3. After a simulation, ALWAYS use 'summarize_last_result' to interpret the data.
"""

# Δημιουργία του Agent
agent_executor = create_react_agent(llm, tools)