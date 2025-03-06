# Using this you agree to the Radlex License
# SEE: https://www.rsna.org/uploadedFiles/RSNA/Content/Informatics/RadLex_License_Agreement_and_Terms_of_Use_V2_Final.pdf
#

import owlready2 as owl
import os
import sys
from contextlib import contextmanager,redirect_stderr
from os import devnull

from ontology.ontology_manager import OntologyManager, RadLexGraphBuilder, ClassesOperations as CO
from settings import RADLEX_DATA, RADLEX_GRAPH_DIR, RADLEX_GRAPH

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

        # Load the ontology, suppressing error outputs:
        # Radlex contains cycles that owlready outputs as Warnings
        with suppress_stderr():
            builder.build_graph()

        isolated_nodes = [node for node in builder.graph.nodes() if builder.graph.degree(node) == 0]
        print(f"🔎 Check Nodi isolati: {len(isolated_nodes)}")

        builder.prune_graph()
        print(builder.count_subgraph_sizes())

        builder.save_graph()

    return builder.graph