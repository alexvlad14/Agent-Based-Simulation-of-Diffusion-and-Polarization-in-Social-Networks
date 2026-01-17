import sys
import os
import time

# Προσθήκη φακέλου src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from langchain_core.messages import HumanMessage, SystemMessage
from agent import agent_executor, system_prompt

def main():
    # --- ΛΙΣΤΑ ΕΝΤΟΛΩΝ (Όλα τα Tasks με τη σειρά) ---
    queries = [
        # === TASK 1: Data Loading ===
        "Load the Facebook graph from ./data/facebook_combined.txt and confirm nodes/edges.",
        
        # === TASK 3: Agent Tools (Basic Graph Queries) ===
        "Check the neighbors of node '0'. How many are there and list the first 5.",
        "Find the shortest path between node '0' and node '2000'. Show me the path sequence.",
        "Identify the most important nodes based on degree centrality. List the top 5.",
        "Detect communities in the graph using the available tool and report the number of communities found.",

        # === TASK 4 & 5: Simulation Scenarios & Evaluation ===
        # Σενάριο A: Diffusion (Influencers vs Random) - Με τις "καλές" παραμέτρους
        "Run influencers vs random diffusion with k=50, T=30, alpha=0.6, thr=0.35. Make sure to compare the final reach percentages.",
        
        # Σενάριο B: Polarization (Emergent Behavior)
        "Run polarized communities diffusion setting parameter frac to 0.10 (10 percent). Use T=30, thr=0.4."
    ]

    print("--- STARTING FULL SNA PROJECT SIMULATION ---")

    # Αρχικοποίηση μνήμης με το System Prompt
    chat_history = [SystemMessage(content=system_prompt)]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔹 Step {i}: {query}")
        
        # Προσθήκη της ερώτησης
        chat_history.append(HumanMessage(content=query))
        
        try:
            start_time = time.time()
            
            # Εκτέλεση Agent
            result = agent_executor.invoke({"messages": chat_history})
            
            # Λήψη απάντησης
            ai_response = result["messages"][-1]
            print(f"🔸 Agent: {ai_response.content}")
            
            # Ενημέρωση ιστορικού (για να θυμάται τα προηγούμενα)
            chat_history = result["messages"]
            
            print(f"   (Time taken: {time.time() - start_time:.2f}s)")
            
        except Exception as e:
            print(f"❌ Error in step {i}: {e}")

    print("\n--- SIMULATION COMPLETED ---")

if __name__ == "__main__":
    main()