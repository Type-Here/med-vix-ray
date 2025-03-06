# Using this you agree to the Radlex License
# SEE: https://www.rsna.org/uploadedFiles/RSNA/Content/Informatics/RadLex_License_Agreement_and_Terms_of_Use_V2_Final.pdf
#

import owlready2 as owl
import networkx as nx
import os, json, sys
from contextlib import contextmanager,redirect_stderr
from os import devnull

from ontology.ontology_manager import OntologyManager, RadLexGraphBuilder, ClassesOperations as CO
from settings import RADLEX_DATA, RADLEX_GRAPH_DIR, RADLEX_GRAPH, FILTER_RADLEX_JSON

@contextmanager
def suppress_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err

# Add the settings.py directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

mimic_labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
]


class_filter = {
    "anatomical entity": CO.CHECK_LEMMA,
    "clinical finding": CO.CHECK_LEMMA,
    "imaging observation": ["asymmetry", "lesion", "lung imaging observation", "normal anatomy", "visible anatomic entity", CO.CHECK_KEY],
    "imaging sign": CO.CHECK_DEFINITION,
    "imaging specialty": CO.SKIP,
    "non-anatomical substance": CO.SKIP,
    "object": CO.KEEP_ALL,
    "procedure": ["access procedure", "opacity"],
    "procedure step": CO.SKIP,
    "process": CO.SKIP,
    "property": [CO.CHECK_KEY, "imaging error"],
    "imaging procedure property": "radiography procedure attribute",
    "radlex descriptor": ["aggressiveness descriptor", CO.CHECK_LEMMA],
    "report component": CO.KEEP_ALL,
    "temporal entity": CO.SKIP
}

relevant_categories = ["anatomical entity", "imaging observation", "finding",
                                    "imaging procedure attribute", "relationship", "pathologic finding", "imaging observation", "anatomical entity"]
root_label = "RID1" # Radlex entity

radlex_entity_children = ["anatomical entity", "data for report", "foreign body", "image quality", "imaging observation",
                          "imaging observation characteristic", "imaging procedure attribute",
                          "imaging service request", "modifier", "procedure step", "relationship",] # skipped substance, teaching attribute, treatment

relevant_anatomy = ["anatomical", "organ", "region", "respiratory", "lung", "pulmonary",
                    "pleura", "thorax", "chest", "mediastinum", "trachea", "bronchus",
                    "heart", "cardiac", "card", "diaphragm", "rib", "spine", "clavicle",
                    "sternum", "aorta", "pulmonary artery", "pulmonary vein", "mediastinum", "thymus",
                    "lung lobe", "lung segment", "lung base", "lung apex", "lung hilum", "lung fissure",
                    "alveoli", "bronchioles", "vertebra", "intercostal"]

other_keywords = ["pneumonia", "pneumothorax", "pleural effusion", "atelectasis", "lung opacity",
                  "lung lesion", "lung mass", "ischemia", "cough", "symptom", "pathophysiologic",
                  "disease", "injury", "fracture", "neoplastic", "valvular", "fracture", "radiology",
                  "opacity", "consolidation", "edema",  "enlarged", "radiolucent", "radiopaque",
                  "support devices", "glass", "neoplastic", "finding", "no finding"]

#chester = onto.search_one(iri="*RID3852")
#print(f"Chester: {chester}")
#print(f"Chester subClass of: {ontology_manager.get_is_subclass_of(chester)}")
#print(f"Chester subclasses: {list(chester.subclasses())}")

# ======================= EXECUTION =======================

def load_or_create_graph():
    """
    Load the graph if it exists, otherwise create it.
    """
    # Load the ontology, suppressing error outputs
    with suppress_stderr():
        onto = owl.get_ontology(RADLEX_DATA).load()

    ontology_manager = OntologyManager(ontology=onto, obtainable_labels=relevant_categories + other_keywords,
                                       anatomical_labels=relevant_anatomy,
                                       classification_labels=radlex_entity_children)
    builder = RadLexGraphBuilder(ontology_manager, root_label=root_label, class_filter=class_filter)

    if os.path.exists(RADLEX_GRAPH):
        print("🚀 Graph already exists, loading...")
        print(f"🔄 Loading existing graph from {RADLEX_GRAPH_DIR}")
        builder.load_graph()

    else:
        print("🚧 Graph not found, creating a new one...")
        print(f"🛠️ Creating new graph at {RADLEX_GRAPH_DIR}")

        # Carica il JSON (puoi leggerlo da file)
        with open(FILTER_RADLEX_JSON, "r") as f:
            json_structure = json.load(f)

        # Load the ontology, suppressing error outputs:
        # Radlex contains cycles that owlready outputs as Warnings
        with suppress_stderr():
            # builder.build_graph()
            graph_json = builder.create_graph_from_json(json_structure)

        isolated_nodes = [node for node in builder.graph.nodes() if builder.graph.degree(node) == 0]
        print(f"🔎 Check Nodi isolati: {len(isolated_nodes)}")

        #builder.prune_graph()
        #print(builder.count_subgraph_sizes())

        builder.save_graph()

    return builder.graph


def search_nodes_by_label(graph: nx.DiGraph, labels_to_find):
    """
    Searches for nodes in the graph whose labels match the provided search terms.

    Args:
        graph (nx.DiGraph): The directed graph.
        labels_to_find (str or list): The label(s) to search for.

    Returns:
        dict: A dictionary where keys are node IDs and values are their attributes.
    """

    if isinstance(labels_to_find, str):
        labels_to_find = [labels_to_find]  # Convert single string to list

    labels_to_find = {label.lower() for label in labels_to_find}  # Normalize labels to lowercase
    matching_nodes = {}

    # Scan all nodes and check if their label is in the search list
    for node, attributes in graph.nodes(data=True):
        node_label = attributes.get("label", "").strip().lower()

        if node_label and node_label in labels_to_find:
            matching_nodes[node] = attributes

    if matching_nodes:
        print(f"✅ Found {len(matching_nodes)} matching nodes!")
    else:
        print("❌ No matching nodes found.")

    return matching_nodes


# ======================== MAIN =======================
if __name__ == "__main__":

    choice = "9"
    graph_r = None
    match_node = None

    def print_menu():
        print("Options:")
        print("1. Load graph")
        print("2. Search nodes by label")
        print("3. Show children")
        print("4. Graph stats")
        print("9. Show options")
        print("0. Exit")

    print_menu()
    while choice != "0":

        choice = input("\nChoice: ")

        if choice == "1":
            graph_r = load_or_create_graph()
            print("Graph loaded successfully!")

        elif choice == "2":
            if 'graph' not in locals():
                print("❌ Graph not loaded. Please load the graph first.")
                continue

            labels = input("Enter label(s) to search for (comma-separated): ")
            labels_to_retrieve = [label.strip() for label in labels.split(",")]

            matching_nodes = search_nodes_by_label(graph_r, labels_to_retrieve)

            if matching_nodes:
                match_node = matching_nodes[list(matching_nodes.keys())[0]]  # Get the first matching node
                match_node["id"] = list(matching_nodes.keys())[0]  # Add the node ID to the attributes
                for node_id, attributes in matching_nodes.items():
                    print(f"Node ID: {node_id}, Attributes: {attributes}")

        elif choice == "3":
            if match_node is None:
                print("❌ No matching nodes found. Please search for nodes first.")
                continue
            print(match_node)
            children = graph_r.successors(match_node["id"])
            for ch in children:
                print(f"Child Node ID: {ch}, Attributes: {graph_r.nodes[ch]}")

        elif choice == "4":
            if graph_r is None:
                print("❌ Graph not loaded. Please load the graph first.")
                continue
            print("Graph Stats:")
            print("Number of nodes:", graph_r.number_of_nodes())
            print("Number of edges:", graph_r.number_of_edges())
            print("Graph density:", nx.density(graph_r))
            print("Is the graph directed?", graph_r.is_directed())
            print("Is the graph connected?", nx.is_connected(graph_r.to_undirected()))
            isolated_nodes = [node for node in graph_r.nodes() if graph_r.degree(node) == 0]
            print(f"🔎 Check Nodi isolati: {len(isolated_nodes)}\n")
            root_node = next(node for node, attr in graph_r.nodes(data=True) if attr.get("type") == "Root")
            print("Root node:", root_node, "Attributes:", graph_r.nodes[root_node])
            print("Children of root node:")
            for child in graph_r.successors(root_node):
                print(f"Child Node ID: {child}, Attributes: {graph_r.nodes[child]}")
                number_of_children = graph_r.out_degree(child)
                print(f"Number of children for {child}: {number_of_children}")
                descendants = nx.descendants(graph_r, child)
                print(f"Descendants of {child}: {len(descendants)}")

            # Get number of different types of relationships
            relationship_types = set()
            for _, _, data in graph_r.edges(data=True):
                relationship_types.add(data["relation"])
            print("\n-Different types of relationships:", len(relationship_types), "; ", relationship_types)

            # Get number of different types of nodes
            node_types = set()
            for _, data in graph_r.nodes(data=True):
                node_types.add(data["type"])
            print("\n-Different types of nodes:", len(node_types), "; ", node_types)
            # Get number of different anatomical entities

        elif choice == "9":
            print_menu()

        elif choice == "0":
            print("🚪 Exiting...")
            break

        else:
            print("❌ Invalid choice. Please try again.")

