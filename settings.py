import os
RADLEX_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology', 'data')
RADLEX_DATA = os.path.join(RADLEX_DATA_DIR, 'RadLex.owl')

RADLEX_GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology')
RADLEX_GRAPH = os.path.join(RADLEX_GRAPH_DIR, 'radlex_graph.json')

FILTER_RADLEX_JSON = os.path.join(RADLEX_DATA_DIR, 'filter.json')

# Report data
MIMIC_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic')

# Dataset path
DATASET_PATH = os.environ.get('MIMIC_DATASET_PATH')

# Directory where csv files are stored, containing the dataset information
DATASET_INFO_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic', 'info')

# ====== PARAMETERS FOR FEATURE EXTRACTION IN NLP (Graph) ======

NUM_WORKERS = 8 # Number of parallel workers for processing (Threads)

# ======================== Hyperparameters for TRAINING =================================

NUM_EPOCHS = 10
LEARNING_RATE_TRANSFORMER = 1e-5 # For Transformer Blocks
LEARNING_RATE_CLASSIFIER = 1e-4 # For Classifier Head
BATCH_SIZE = 16

UNBLOCKED_LEVELS = 3 # Number of unblocked levels in the Swin Transformer (from end to start)

SWIN_MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'swin_model.pth')
SWIN_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'swin_stats.json')

# ====== PARAMETERS FOR FEATURE EXTRACTION IN XAI (Attention Map) ======

ATTENTION_MAP_THRESHOLD = 0.5 # Threshold for attention map analysis


# ===== PARAMETERS FOR SIMILARITY CALCULATION ======

# Threshold for similarity calculation between graph nodes and medical reports
SIMILARITY_THRESHOLD = 0.7