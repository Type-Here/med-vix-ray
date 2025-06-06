import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def show_graph(graph):
    """
    Function to display the graph.
    Uses pygraphviz for hierarchical layout if available.
    Args:
        graph (nx.Graph): The graph to be displayed.
    """

    # Verify if pygraphviz is available for hierarchical layout
    try:
        pos = graphviz_layout(graph, prog="dot")  # Tree layout
    except ImportError:
        pos = nx.spring_layout(graph)  # If pygraphviz is not available, use spring layout

    plt.figure(figsize=(150, 50))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color="lightblue",
            edge_color="gray", font_size=10, font_weight="bold", arrows=True)
    plt.title("Visualizzazione Gerarchica del Grafo a partire da RID1")
    plt.show()