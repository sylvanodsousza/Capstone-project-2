import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def run_network():
    print("🌍 Running Network Analysis...\n")

    # Ensure folders exist
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    # Load cleaned geo-enriched data
    df = pd.read_csv("outputs/processed/incidents_geo.csv")

    # ----------------------------------------------------
    # 1️⃣ BUILD ORIGIN → DESTINATION EDGES
    # ----------------------------------------------------
    print("🔗 Building trafficking edges...")

    # Filter rows with valid origin & destination
    edges = df[
        df["origin_country"].notna() & df["destination_country"].notna()
    ][["origin_country", "destination_country"]]

    # Count route frequency
    edge_counts = (
        edges.groupby(["origin_country", "destination_country"])
        .size()
        .reset_index(name="count")
    )

    edge_counts.to_csv("outputs/tables/route_edges.csv", index=False)

    print(f"✔ Route edges saved: {edge_counts.shape[0]} routes\n")

    # ----------------------------------------------------
    # 2️⃣ BUILD NETWORK GRAPH
    # ----------------------------------------------------
    print("🌐 Constructing graph...\n")

    G = nx.from_pandas_edgelist(
        edge_counts,
        source="origin_country",
        target="destination_country",
        edge_attr="count",
        create_using=nx.DiGraph()
    )

    # ----------------------------------------------------
    # 3️⃣ CENTRALITY ANALYSIS (CHOKEPOINTS)
    # ----------------------------------------------------
    print("🔥 Calculating chokepoints (betweenness centrality)...")

    centrality = nx.betweenness_centrality(G, weight="count", normalized=True)

    centrality_df = (
        pd.DataFrame(centrality.items(), columns=["country", "centrality"])
        .sort_values("centrality", ascending=False)
    )

    centrality_df.to_csv("outputs/tables/chokepoints.csv", index=False)

    print(f"✔ Chokepoints saved: {centrality_df.shape[0]} countries\n")

    # ----------------------------------------------------
    # 4️⃣ PLOT NETWORK GRAPH
    # ----------------------------------------------------
    print("🖼️ Plotting network graph...")

    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Node size based on centrality
    node_sizes = [5000 * centrality.get(node, 0.001) for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Global Wildlife Trafficking Network\n(Node size = Chokepoint Importance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/plots/network_graph.png", dpi=300)
    plt.close()

    print("✔ Network graph saved.\n")
    print("🎉 Network Analysis complete!")

if __name__ == "__main__":
    run_network()
