from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()



def __tokenize_label(label):
    """
    Tokenize a label into a list of keywords.

    Args:
        label (str): text to be tokenized.

    Returns:
        list: List of keywords with stemming applied.
    """
    if not label:
        return None

    label = label.lower()
    tokens = re.findall(r'\b\w+\b', label)  # Extract words, ignoring punctuation
    return tokens


def match_label(label_tokens, word_list):
    """
    Checks if at least one token from `label_tokens` is present in `word_list`.

    Args:
        label_tokens (list): List of tokenized label words.
        word_list (list): List of keywords to match against.

    Returns:
        bool: True if at least one term from `label_tokens` is present in `word_list`, False otherwise.
    """
    return any(token in word_list for token in label_tokens)





def tokenize_and_stem_list(word_list):
    """
    Tokenize and stem the words in list to recognize similar variants.
    Args:
        word_list (list or str): List of words to be tokenized and stemmed or a single string.
    Returns:
        list: List of stemmed tokens.
    """
    if not word_list:
        return None

    tokenized_list = set()
    if isinstance(word_list, str):
        word_list = [word_list]

    for word in word_list:
        tokens = __tokenize_label(word)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]  # Apply stemming
        tokenized_list.update(stemmed_tokens)

    return list(tokenized_list)


# ===== EXAMPLE =====
# word_list = ["Myocarditis", "Pericarditis", "Cardiomegaly"]
# tokenized_stemmed_words = tokenize_and_stem(word_list)

# print(f"Lista tokenizzata con stemming: {tokenized_stemmed_words}")
