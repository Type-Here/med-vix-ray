import pandas as pd
import networkx as nx
import tokenizer as tk
from enum import Enum
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

_relative_labels = { "subclass_of": "subClassOf", "parent_of": "parent_of"}


# Enum for Classes Operations in relevance
class ClassesOperations(Enum):
    CHECK_KEY = 1 # Check if the class is in the filter keys
    CHECK_LEMMA = 2 # Check if the class is in the tokenized and stemmed list
    CHECK_DEFINITION = 3 # Check if class label or definition contains at least one keyword in tokenized and stemmed list
    SKIP = 4 # Skip the class
    KEEP_ALL = 5 # Keep all classes
    CHECK_VALUES = 6 # Check if class label is in filter values


# relationships = ["Has_finding", "Has_location", "May_Cause", "Origin_of", "Member_of", "Has_member", "Has_Subtype"]


class OntologyManager:
    """
        Class to manage and extract data from an ontology.
    """

    def __init__(self, ontology: Ontology, obtainable_labels: list = None,
                 anatomical_labels: list = None, classification_labels: list = None):
        """
        Initializes the OntologyManager with the given ontology and optional parameters.
        :param ontology: Ontology object.
        :param obtainable_labels: List of labels to be obtained
        :param anatomical_labels: List of anatomical labels to be considered
        """
        super().__init__()
        self.onto = ontology
        self.obtainable_labels = tk.tokenize_and_stem_list(obtainable_labels) or []
        self.anatomical_labels = tk.tokenize_and_stem_list(anatomical_labels) or []
        self.classification_labels = tk.tokenize_and_stem_list(classification_labels) or []

        self.relevant_list = self.obtainable_labels + self.anatomical_labels

        print(f"✅ OntologyManager initialized with \n Obtainable labels:{self.obtainable_labels[:5]};")
        print(f" \n Anatomical labels: {self.anatomical_labels[:5]};"
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


# ================================ GRAPH BUILDER ==================================


class RadLexGraphBuilder:
    """
        Class to build a graph from the RadLex ontology.
    """

    def __init__(self, ontology_manager: OntologyManager, class_filter = None, root_label="RadLex entity"):
        """
        Init Ontology and create a directed graph.
        Args:
            ontology_manager (OntologyManager): OntologyManager object.
            class_filter (dict): Dictionary to filter classes.
            root_label (str): Root label to start the graph building.
        """
        self.ontology_manager = ontology_manager
        self.onto = self.ontology_manager.onto
        self.root_label = root_label  # Define root label

        self.class_filter = class_filter or {}
        self.graph = nx.DiGraph()  # Directed graph



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

    def is_relevant_entity(self, cls, root_label="RadLex entity", check_definition = False):
        """
        Check if the class belongs to one of the relevant categories.
        Args:
            cls (Thing): Class to check.
            root_label (str): Root label to check against.
                If the class is a subclass of this label, check for relevance
                in classification_labels instead (if any).
            check_definition (bool): If True, check if the class label or definition. If False, check only the label.
        Returns:
            bool: True if the class is relevant, False otherwise.
        """
        # Check if the class is a subclass of the root label
        subclass = self.ontology_manager.get_is_subclass_of(cls)
        if subclass and subclass == root_label and self.ontology_manager.classification_labels:
            # Check if the class belongs to one of the classification labels after tokenization
            return any(cat in tk.tokenize_and_stem_list(self.get_property(cls, _prefLabel))
                       for cat in self.ontology_manager.classification_labels)

        # Check: if the class is a leaf node or has no subclasses, check if it belongs to anatomical or obtainable labels
        # elif cls.subclasses() is None or not list(cls.subclasses()):
        else:
            pref_label = tk.tokenize_and_stem_list(self.get_property(cls, _prefLabel))
            return any(cat in pref_label for cat in self.ontology_manager.relevant_list) or (
                check_definition and any(cat in tk.tokenize_and_stem_list(self.get_property(cls, _definition))
                                                            for cat in self.ontology_manager.relevant_list)
            )

        # return True

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
                # Add node if it doesn't exist
                if related_rid not in self.graph:
                    attributes = self.ontology_manager.extract_data(related_cls, list(_properties_label_map.keys()))
                    self.graph.add_node(related_rid, **attributes)
                # Add edge
                self.graph.add_edge(source.name, related_rid, relation=prop)

    def __add_edge_to_children(self, cls):
        """
        Adds edges to the graph for all subclasses of the given class.
        :param cls (ThingClass): Class to add edges for.
        """
        for subclass in cls.subclasses():
            if subclass.name in self.graph:
                self.graph.add_edge(cls.name, subclass.name, relation="parent_of")

    def __add_node(self, cls):
        """
        Adds a node to the graph if it doesn't exist.
        Args:
            cls (Thing): Class to add.
        """
        rid = cls.name
        if rid not in self.graph:
            # Extract attributes
            attributes = self.ontology_manager.extract_data(cls, list(_properties_label_map.keys()))
            # Add node to the graph
            self.graph.add_node(rid, **attributes)

    def __add_node_with_hierarchy(self, cls, parent=None, operation=ClassesOperations.CHECK_KEY, keyword=None):
        """
        Adds a node to the graph if it doesn't exist, and recursively connects its children.
        Args:
            cls (Thing): Class to add.
            parent (str): Parent class ID.
        """
        relevant_subclasses = []
        rid = cls.name
        name = self.get_property(cls, _prefLabel)
        is_relevant = False
        subclasses = cls.subclasses()

        # print(f"-For {name.lower()} ")
        if operation == ClassesOperations.CHECK_LEMMA:
            is_relevant = self.is_relevant_entity(cls, parent, check_definition = False)
            relevant_subclasses = subclasses

        elif operation == ClassesOperations.CHECK_DEFINITION:
            is_relevant = self.is_relevant_entity(cls, parent, check_definition = True)
            relevant_subclasses = subclasses

        elif operation == ClassesOperations.CHECK_KEY:
            values = self.class_filter.get(name.lower(), None)
            # print("- Check-Key")
            # print(f"Before: -Op: {operation.name}; - is_relevant: {is_relevant}; -Subcls: {subclasses};")
            if not values:
                return
            elif isinstance(values, Enum):
                if values == ClassesOperations.SKIP:
                    return
                operation = values
                relevant_subclasses = cls.subclasses()
            else:
                operation = ClassesOperations.CHECK_VALUES
                relevant_subclasses = [sub for sub in cls.subclasses()
                                       if self.get_property(sub, _prefLabel).lower() in values]

            is_relevant = True

        elif operation == ClassesOperations.KEEP_ALL:
            is_relevant = True
            relevant_subclasses = subclasses

        elif operation == ClassesOperations.CHECK_VALUES:
            values = self.class_filter.get(keyword)
            if isinstance(values, list):
                enum_values = [value for value in values if isinstance(value, Enum)]
                values = [value for value in values if not isinstance(value, Enum)]
                is_relevant = any(name.lower() == value.lower() for value in values)
                relevant_subclasses = [sub for sub in cls.subclasses() if self.get_property(sub, _prefLabel).lower() in values]
                operation = ClassesOperations.KEEP_ALL

                for enum_value in enum_values:
                    if enum_value == ClassesOperations.CHECK_KEY:
                        children_in_key = [sub for sub in subclasses if sub in self.class_filter.keys()]
                        relevant_subclasses.extend(children_in_key)
                    elif enum_value == ClassesOperations.CHECK_LEMMA:
                        children_in_lemma = [sub for sub in subclasses if self.is_relevant_entity(sub, name)]
                        relevant_subclasses.extend(children_in_lemma)

            elif isinstance(values, Enum):
                operation = values or ClassesOperations.SKIP
                is_relevant = True

                if values == ClassesOperations.CHECK_KEY:
                    children_in_key = [sub for sub in subclasses if sub in self.class_filter.keys()]
                    relevant_subclasses = children_in_key

                elif values == ClassesOperations.CHECK_LEMMA:
                    children_in_lemma = [sub for sub in subclasses if self.is_relevant_entity(sub, name)]
                    relevant_subclasses = children_in_lemma

        elif operation == ClassesOperations.SKIP:
            return

        # print(f" After: -Op: {operation.name}; - is_relevant: {is_relevant}; -Subcls: {subclasses};\n")

        if is_relevant:
            # Add node if it doesn't exist
            self.__add_node(cls)
            # Add relationships
            self.__add_edge_from_attributes(cls)


            # Scan for subclasses recursively
            for sbc in relevant_subclasses:
                if isinstance(sbc, Enum):
                    operation = ClassesOperations.CHECK_KEY
                self.__add_node_with_hierarchy(sbc, parent=rid, operation=operation, keyword=name.lower())

        if parent:
            self.graph.add_edge(parent, rid, relation="parent_of")

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
        print("Radlex Entity Children:")
        for cls in radlex_entity.subclasses():
            print(f"- {cls.name}, {self.get_property(cls, _prefLabel)}")
            self.__add_node_with_hierarchy(cls, operation=ClassesOperations.CHECK_KEY, parent=radlex_entity.name)
            #self.graph.add_edge(radlex_entity.name, cls.name, relation="parent_of")

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

    def prune_graph(self, valid_labels = None):
        """
        Prunes the directed graph by removing leaf nodes that:
        1. Do not have a label.
        2. Have a label not present in the valid_labels (already tokenized).
        3. If a node becomes a leaf and is not valid, remove it iteratively up to the first common ancestor.

        Args:
            valid_labels (list): Set of valid labels. **Attention**: If None, will be used self.relevant_list.

        Returns:
            nx.DiGraph: Pruned graph.
        """

        if valid_labels is None:
            valid_labels = self.ontology_manager.relevant_list

        def is_valid(node):
            """Check if the node is valid based on its label."""
            label = self.graph.nodes[node].get("label", "")
            tokenized_label = tk.tokenize_and_stem_list(label)
            return bool(label) and any(token in tokenized_label for token in valid_labels)

        removed_nodes = set()
        while True:
            # Identify leaf nodes (nodes with no outgoing edges)
            leaves = {node for node in self.graph.nodes if self.graph.out_degree(node) == 0}

            # Find leaves to remove
            to_remove = {leaf for leaf in leaves if not is_valid(leaf)}

            # If no more nodes to remove, stop
            if not to_remove:
                break

            # Remove nodes
            self.graph.remove_nodes_from(to_remove)
            removed_nodes.update(to_remove)

        print(f"✅ Pruning complete! Removed {len(removed_nodes)} nodes.")
        print(f"✅ Remaining nodes: {self.graph.number_of_nodes()}")

        return self.graph

    import networkx as nx

    def count_subgraph_sizes(self, parent_relation="parent_of"):
        """
        Prints the number of nodes in each subgraph rooted at the first-level children of the given root node,
        following the specified parent-child relationship.

        Args:
            parent_relation (str): The relationship label to follow for parent-child connections.

        Returns:
            dict: A dictionary with first-level child nodes as keys and their subgraph size as values.
        """
        root = self.root_label
        graph = self.graph

        if root not in graph:
            print(f"❌ Root node '{root}' not found in the graph!")
            return {}

        # Find first-level children of the root based on the "parent_of" relationship
        first_level_children = [
            child for child in graph.successors(root) if graph.edges[root, child].get("relation") == parent_relation
        ]

        subgraph_sizes = {}

        print(f"📌 Root node: {root}")
        print(f"📌 First-level children: {first_level_children}\n")

        for child in first_level_children:
            # Get all descendants of this child (nodes reachable from it)
            descendants = nx.descendants(graph, child)
            subgraph_size = len(descendants) + 1  # Include the child itself

            info_node = graph.nodes[child]

            subgraph_sizes[child] = subgraph_size
            print(f"🔹 Subgraph rooted at '{child}' - {info_node['label']} → {subgraph_size} nodes")

        return subgraph_sizes
