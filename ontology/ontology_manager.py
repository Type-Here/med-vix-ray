import pandas as pd
import networkx as nx
import tokenizer as tk
import json

from owlready2 import Ontology, ThingClass

_prefLabel = "preferred_label"
_definition = "definition"
_subclassOf = "subclass_of"
_synonym = "synonym"

# Properties Label Map
_properties_label_map = {
    "preferred_label": ["Preferred_name", "label"],
    "definition": ["Definition"],
    "synonym": ["Synonym"],
}

_relationship_edges ={
    "cause" : "May_Cause",
    "be_caused": "May_Be_Caused_By",
    "origin" : "Origin_of",
    "member" : "Has_Member",
}

__relative_labels = { "subclass_of": "subClassOf", "parent_of": "parent_of"}

#relationships = ["Has_finding", "Has_location", "May_Cause", "Origin_of", "Member_of", "Has_member", "Has_Subtype"]


class OntologyManager:
    """
        Class to manage and extract data from an ontology.
    """

    def __init__(self, ontology: Ontology, obtainable_labels: list = None,
                 exclude_labels: list = None, anatomical_labels: list = None, classification_labels: list = None):
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
        self.exclude_labels = tk.tokenize_and_stem_list(exclude_labels) or []
        self.anatomical_labels = tk.tokenize_and_stem_list(anatomical_labels) or []
        self.classification_labels = tk.tokenize_and_stem_list(classification_labels) or []
        self.relevant_list = self.obtainable_labels + self.anatomical_labels

        print(f"✅ OntologyManager initialized with \n Obtainable labels:{self.obtainable_labels[:5]};")
        print(f" \n Exclude Labels: {self.exclude_labels[:5]};"
              f" \n Anatomical labels: {self.anatomical_labels[:5]};"
              f"\n Classification Labels: {self.classification_labels[:5]};")

    def get_classes(self):
        """
        Returns:
            list: A list of all classes in the ontology.
        """
        return list(self.onto.classes())

    def get_property(self, cls, property_name_list, single_value=True):
        """
        Get the property of a class. Check if the class has the properties
        defined in the property_name_list and return the first one found.
        Args:
            cls (ThingClass or str): Class to retrieve properties from or
            class name (iri) to retrieve properties from.
            property_name_list (list[str]): List of property names to retrieve.
            single_value (bool): If True, return the first value found, otherwise return all values.
        Returns:
            str: The value of the first property found or None if not found.
        """

        # If cls is an instance of Thing get the label from list
        if isinstance(cls, ThingClass):
            for annotation in self.onto.annotation_properties():
                value = getattr(cls, annotation.name, None)
                # Check if the property name from annotation is in the list
                if value and any(label in annotation.name for label in property_name_list):
                    return str(value[0]) if single_value else [str(v) for v in value]  # Convert locstr to string or list of strings
                    # return str(value[0])  # Convert locstr to string
            if "Preferred_name" in property_name_list:
                return cls.name
            return None  # Default: return None if no property is found

        elif isinstance(cls, str):
            # Search for the class by name
            cls = self.onto.search_one(iri=f"*{cls}")
            if cls:
                # Recursively call the function
                return self.get_property(cls, property_name_list)
            if "Preferred_name" in property_name_list:
                return cls.name
            return None

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

    def extract_data(self, cls, keywords_list):
        """
            Extracts data from an ontology individual.
            Args:
                cls (Thing): Class to extract data from.
                keywords_list (list): List of keywords to extract data from. Keywords must be in _properties_label_map.
            Returns:
                dict: Extracted data.
        """
        pref_label = self.get_property(cls, _properties_label_map[_prefLabel])
        node_type = cls.is_a[0].name if cls.is_a else "Unknown"

        data = {}
        for label in keywords_list:
            value = self.get_property(cls, _properties_label_map[label])
            data[label] = value or ""
        data.update({"label": pref_label, "type": node_type})
        return data


class RadLexGraphBuilder:
    """
        Class to build a graph from the RadLex ontology.
    """

    def __init__(self, ontology_manager: OntologyManager, root_label="RadLex entity"):
        """
        Init Ontology and create a directed graph.
        """
        self.ontology_manager = ontology_manager
        self.onto = self.ontology_manager.onto
        self.root_label = root_label  # Define root label

        self.graph = nx.DiGraph()  # Directed graph
        self.keywords_list = _properties_label_map.keys()


    def get_property(self, cls, property_key):
        """
        Get the property of a class.
        Check if the class has the properties defined in the property_name_list and return the first one found.
        Args:
            cls (ThingClass or str): Class to retrieve properties from or class name (iri) to retrieve properties from.
            property_key: Property name to retrieve as key defined in self.properties_label_map.
        Returns:
            str: The value of the first property found or None if not found.
        See Also:
            OntologyManager.get_property
        """
        instance = _properties_label_map.get(property_key)
        return self.ontology_manager.get_property(cls, instance)

    def is_relevant_entity(self, cls, root_label="RadLex entity"):
        """
        Check if the class belongs to one of the relevant categories.
        Args:
            cls (Thing): Class to check.
            root_label (str): Root label to check against.
                If the class is a subclass of this label, check for relevance
                in classification_labels instead (if any).
        Returns:
            bool: True if the class is relevant, False otherwise.
        """
        # Check if the class is a subclass of the root label
        subclass = self.ontology_manager.get_is_subclass_of(cls)
        if subclass and subclass == root_label and self.ontology_manager.classification_labels:
            # Check if the class belongs to one of the classification labels after tokenization
            return any(cat in tk.tokenize_label(self.get_property(cls, _prefLabel))
                       for cat in self.ontology_manager.classification_labels)

        # Check: if the class is a leaf node or has no subclasses, check if it belongs to anatomical or obtainable labels
        elif cls.subclasses() is None or not list(cls.subclasses()):
            pref_label = tk.tokenize_label(self.get_property(cls, _prefLabel))
            return any(cat in pref_label for cat in self.ontology_manager.relevant_list)

        return True

    def __add_edge_from_attributes(self, source):
        """
        Adds an edge to the graph if it doesn't exist.
        Args:
            source (ThingClass): Source node.
        """
        # Add relationships
        for prop in _relationship_edges.values():
            related_classes = getattr(source, prop, [])
            for related_cls in related_classes:
                related_rid = related_cls.name
                related_label = self.get_property(related_cls, _prefLabel)
                if related_rid not in self.graph:
                    attributes = self.ontology_manager.extract_data(related_cls, list(_properties_label_map.keys()))
                    self.graph.add_node(related_rid, **attributes)
                self.graph.add_edge(source.name, related_rid, relation=prop)

    def __add_edge_to_children(self, cls):
        for subclass in cls.subclasses():
            if subclass.name in self.graph:
                self.graph.add_edge(cls.name, subclass.name, relation="parent_of")

    def __add_node_with_hierarchy(self, cls, parent=None):
        """
        Adds a node to the graph if it doesn't exist, and recursively connects its children.
        Args:
            cls (Thing): Class to add.
            parent (str): Parent class ID.
        """
        rid = cls.name

        if rid not in self.graph:
            # Extract attributes
            attributes = self.ontology_manager.extract_data(cls, list(_properties_label_map.keys()))
            # Add node to the graph
            self.graph.add_node(rid, **attributes)

        #if parent:
        #    self.graph.add_edge(parent, rid, relation="subclass_of")

        # Scan for subclasses recursively
        for subclass in cls.subclasses():
            if self.is_relevant_entity(subclass):
                self.__add_node_with_hierarchy(subclass, rid)
                self.graph.add_edge(rid, subclass.name, relation="parent_of")
                # Add relationships
                self.__add_edge_from_attributes(cls)

        # Add Parent to Children Edges
        #self.__add_edge_from_attributes(cls)

    def build_graph(self):
        """
        Builds the graph by exploring the hierarchy under 'RadLex entity' using subclasses().
        """
        radlex_entity = self.onto.search_one(iri="*RID1")  # 'RadLex entity'

        if not radlex_entity:
            print("❌ Errore: 'RadLex entity' non trovato.")
            return

        print("✅ RadLex entity trovato! Scansionando sottoclassi...")
        self.graph.add_node(radlex_entity.name, label=self.get_property(radlex_entity, _prefLabel), type="Root")
        # Starts from RadLex entity and scans subclasses
        for cls in radlex_entity.subclasses():
            if self.is_relevant_entity(cls):
                self.__add_node_with_hierarchy(cls)
                self.graph.add_edge(radlex_entity.name, cls.name, relation="parent_of")

        print(f"✅ Nodi trovati: {self.graph.number_of_nodes()}, "
              f"Archi trovati: {self.graph.number_of_edges()}")

    def save_graph(self, json_path="radlex_graph.json",
                   csv_path="radlex_graph.csv", graphml_path="radlex_graph.graphml"):
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
