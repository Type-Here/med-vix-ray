import spacy
from gensim.models import KeyedVectors
from difflib import get_close_matches
from settings import SIMILARITY_THRESHOLD

# ========================================================== NODES VALUES ATTRIBUTION =========================================================

# 1. Load the pre-trained SciSpacy model for the medical domain
nlp = spacy.load("en_core_sci_sm")

# 2. Load a pre-trained Word2Vec or FastText model to find synonyms
word_vectors = KeyedVectors.load_word2vec_format("word2vec_healthcare.bin", binary=True)


def extract_keywords(text, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Extract keywords from a medical report using SciSpacy + Word2Vec.

    Args:
        text (str): Medical report to analyze.
        similarity_threshold (float): Threshold for synonym similarity.

    Returns:
        list: List of keywords extracted from the report.
    """
    doc = nlp(text)
    keywords = set()

    # Named Entity Recognition (NER) with SciSpacy
    for ent in doc.ents:
        keywords.add(ent.text.lower())

    # If a word is not found, look for synonyms with Word2Vec
    for token in doc:
        if token.text.lower() not in keywords and token.text in word_vectors:
            similar_words = word_vectors.most_similar(token.text, topn=3)
            for word, similarity in similar_words:
                if similarity > similarity_threshold:  # Similarity threshold
                    keywords.add(word.lower())

    return list(keywords)

def match_keywords_to_graph(keywords, graph_json):
    """
    Maps keywords extracted from the report to nodes in the graph.
    Args:
        keywords (list): List of keywords extracted from the report.
        graph_json (dict): JSON graph data.
    Returns:
        dict: Mapping of keywords to graph nodes.
    """
    matched_nodes = {}

    for keyword in keywords:
        # Here we use difflib to find the closest match
        # Because in training use of nlp would slow down the process excessively
        for node in graph_json["nodes"]:
            possible_matches = [node["label"]] + node.get("synonyms", [])
            closest = get_close_matches(keyword, possible_matches, n=1, cutoff=0.6)

            if closest:
                matched_nodes[keyword] = node["label"]

    return matched_nodes




# ========================================================== EDGES VALUES ATTRIBUTION =========================================================

