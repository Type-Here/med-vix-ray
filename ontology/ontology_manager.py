import pandas as pd
import networkx as nx
import json

from owlready2 import Ontology, ThingClass

class OntologyManager:
    """
        Class to manage and extract data from an ontology.
    """
    def __init__(self, ontology: Ontology, obtainable_labels: list = None,
                 exclude_labels: list = None, anatomical_labels: list = None):
        """
        Initializes the OntologyManager with the given ontology and optional parameters.
        :param ontology: Ontology object.
        :param obtainable_labels: List of labels to be obtained
        :param exclude_labels: List of labels to be excluded
        :param anatomical_labels: List of anatomical labels to be considered
        """
        super().__init__()
        self.onto = ontology
        self.obtainable_labels = obtainable_labels or []
        self.exclude_labels = exclude_labels or []
        self.anatomical_labels = anatomical_labels or []

    def get_classes(self):
        """
        Returns:
            list: A list of all classes in the ontology.
        """
        return list(self.onto.classes())

    def get_pref_label(self, cls):
        """ Returns the human-readable name of the class (Preferred_name or alternative).
        Args:
            cls (Thing or str): The class or its name.
        Returns:
            str: The preferred label of the class.

        1. If cls is an instance of Thing, it returns the preferred label.
        2. If cls is a string, it returns the preferred label of the class with that name after searching for it.
        3. If cls is neither, it raises a TypeError.
        4. If the class has no preferred label, it returns the class name.

        Raises:
            TypeError: If cls is neither a Thing nor a string.
        """
        if isinstance(cls, ThingClass):
            # If cls is an instance of Thing get the preferred label
            for annotation in self.onto.annotation_properties():
                value = getattr(cls, annotation.name, None)
                if value and ("Preferred_name" in annotation.name or "label" in annotation.name):
                        return str(value[0])  # Convert locstr to string
            return cls.name # Default: return the name of the class if no preferred label is found

        elif isinstance(cls, str):
            # Search for the class by name
            cls = self.onto.search_one(iri=f"*{cls}")
            if cls:
                # Recursively call the function
                return self.get_pref_label(cls)
            return cls.name

        else:
            raise TypeError("cls must be an instance of Thing or str")

    def get_is_subclass_of(self, cls):
        """ Returns the parent class of the given class.
        Args:
            cls (Thing or str): The class or its name.
        Returns:
            str: The parent class of the given class.
        """
        if isinstance(cls, ThingClass):
            for annotation in self.onto.annotation_properties():
                value = getattr(cls, annotation.name, None)
                if value and ("subClassOf" in annotation.name or "sub_class_of" in annotation.name):
                        return str(value[0])  # Convert locstr to string
            return None

        elif isinstance(cls, str):
            cls = self.onto.search_one(iri=f"*{cls}")
            if cls:
                return self.get_is_subclass_of(cls)
            return None
        else:
            raise TypeError("cls must be an instance of Thing or str")

    def extract_filtered_data(self):
        """ Extracts only relevant data related to obtainable and anatomy label lists. """
        data = []
        for cls in self.get_classes():
            rid = cls.name  # ID RadLex
            pref_label = self.get_pref_label(cls) or "N/A"
            definition = getattr(cls, "definition", ["N/A"])[0]
            parent = cls.is_a[0].name if cls.is_a else "N/A"

            # Filter only for obtainable labels
            if any(label.lower() in pref_label.lower() for label in self.obtainable_labels):
                data.append({
                    "RID": rid,
                    "Preferred_name": pref_label,
                    "Definition": definition,
                    "Parent": parent,
                    "Type": "Pathology"
                })

            # Include anatomical labels
            elif any(term.lower() in pref_label.lower() for term in self.anatomical_labels):
                data.append({
                    "RID": rid,
                    "Preferred_name": pref_label,
                    "Definition": definition,
                    "Parent": parent,
                    "Type": "Anatomical Structure"
                })

        return pd.DataFrame(data)  # Return as DataFrame




class RadLexGraphBuilder:
    def __init__(self, ontology_manager: OntologyManager, root_label="RadLex entity"):
        """
        Init Ontology and create a directed graph.
        """
        self.ontology_manager = ontology_manager
        self.onto = self.ontology_manager.onto
        self.root_label = root_label  # Define root label

        self.graph = nx.DiGraph()  # Directed graph

        # Define categories of interest


    def get_pref_label(self, cls):
        """
        Calls the method from OntologyManager with the same name.
        Returns:
            str: 'preferred_name' of the class. If not found, returns the class name.
        See also:
            - OntologyManager.get_pref_label
        """
        return self.ontology_manager.get_pref_label(cls)

    def is_relevant_entity(self, cls, root_label="RadLex entity"):
        """
        Check if the class belongs to one of the relevant categories.
        Args:
            cls (Thing): Class to check.
        Returns:
            bool: True if the class is relevant, False otherwise.
        """
        subclass = self.ontology_manager.get_is_subclass_of(cls)
        if subclass and subclass == root_label:
            return True

        pref_label = self.get_pref_label(cls).lower()
        return any(cat in pref_label for cat in self.ontology_manager.obtainable_labels + self.ontology_manager.anatomical_labels)

    def add_node_with_hierarchy(self, cls, parent=None):
        """
        Adds a node to the graph if it doesn't exist, and recursively connects its children.
        Args:
            cls (Thing): Class to add.
            parent (str): Parent class ID.
        """
        rid = cls.name
        pref_label = self.get_pref_label(cls)
        node_type = cls.is_a[0].name if cls.is_a else "Unknown"

        if rid not in self.graph:
            self.graph.add_node(rid, label=pref_label, type=node_type)

        if parent:
            self.graph.add_edge(parent, rid, relation="subclass_of")

        # Scan for subclasses recursively
        for subclass in cls.subclasses():
            self.add_node_with_hierarchy(subclass, rid)

    def build_graph(self):
        """
        Builds the graph by exploring the hierarchy under 'RadLex entity' using subclasses().
        """
        radlex_entity = self.onto.search_one(iri="*RID1")  # 'RadLex entity'

        if not radlex_entity:
            print("❌ Errore: 'RadLex entity' non trovato.")
            return

        print("✅ RadLex entity trovato! Scansionando sottoclassi...")

        # Partiamo da RadLex entity e scendiamo nelle sottoclassi
        for cls in radlex_entity.subclasses():
            self.add_node_with_hierarchy(cls)

        print(f"✅ Nodi trovati: {self.graph.number_of_nodes()}, Archi trovati: {self.graph.number_of_edges()}")

    def build_graph_old(self):
        """
        Build the graph filtering only entities under 'RadLex entity'.
        1. Finds the 'RadLex entity' class.
        2. Iterates through its descendants.
        3. Adds nodes and edges for relevant entities.
        4. Filters out irrelevant entities.
        5. Saves the graph in JSON, CSV, and GraphML formats.
        """
        radlex_entity = self.onto.search_one(label=self.root_label)  # Finds the 'RadLex entity'

        if not radlex_entity:
            print(f"❌ Errore: '{self.root_label}' non trovato.")
            return

        print(f"✅ {radlex_entity}  trovato! Costruzione del grafo...")
        print("Inizio a costruire il grafo...")
        print("Discendenti di RadLex entity:")
        print(radlex_entity.descendants())

        # Create nodes and connect only useful relationships
        for cls in radlex_entity.descendants():
            rid = cls.name
            pref_label = self.get_pref_label(cls)
            node_type = cls.is_a[0].name if cls.is_a else "Unknown"

            if self.is_relevant_entity(cls):  # Filter: add only relevant entities
                self.graph.add_node(rid, label=pref_label, type=node_type)

                # Add relationships (has_finding, has_location)
                for prop in ["Has_finding", "Has_location", "May_Cause", "Origin_of", "Member_of", "Has_member"]:
                    if hasattr(cls, prop):
                        for related_cls in getattr(cls, prop):
                            related_rid = related_cls.name
                            related_label = self.get_pref_label(related_cls)
                            if self.is_relevant_entity(related_cls):  # Keep only relevant relationships
                                self.graph.add_node(related_rid, label=related_label, type="Finding/Location")
                                self.graph.add_edge(rid, related_rid, relation=prop)

        print(f"✅ Nodi trovati: {self.graph.number_of_nodes()}, Archi trovati: {self.graph.number_of_edges()}")

    def save_graph(self, json_path="radlex_graph.json", csv_path="radlex_graph.csv", graphml_path="radlex_graph.graphml"):
        """
        Save the graph in JSON, CSV, and GraphML formats.
        Args:
            json_path (str): Path to save JSON file.
            csv_path (str): Path to save CSV file.
            graphml_path (str): Path to save GraphML file.
        """
        # JSON
        graph_data = nx.node_link_data(self.graph)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=4)
        print(f"✅ Grafo salvato in JSON: {json_path}")

        # CSV
        df_edges = pd.DataFrame([(u, v, d["relation"]) for u, v, d in self.graph.edges(data=True)],
                                columns=["Source", "Target", "Relation"])
        df_edges.to_csv(csv_path, index=False)
        print(f"✅ Grafo salvato in CSV: {csv_path}")

        # GraphML
        nx.write_graphml(self.graph, graphml_path)
        print(f"✅ Grafo salvato in GraphML: {graphml_path}")

