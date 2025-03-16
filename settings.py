import os
# Radlex Ontology
RADLEX_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology', 'data')
RADLEX_DATA = os.path.join(RADLEX_DATA_DIR, 'RadLex.owl')

RADLEX_GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology')
RADLEX_GRAPH = os.path.join(RADLEX_GRAPH_DIR, 'radlex_graph.json')

FILTER_RADLEX_JSON = os.path.join(RADLEX_DATA_DIR, 'filter.json')

# Model Save directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models')

# Report data
MIMIC_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic')

# Dataset path
DATASET_PATH = os.environ.get('MIMIC_DATASET_PATH')

# Directory where csv files are stored, containing the dataset information
DATASET_INFO_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic', 'info')
SPLITTED_DATASET_DIR = os.path.join(DATASET_INFO_CSV_DIR, 'split_dataset')

# List of Downloaded Files (if partial_list is not empty) .txt file
DOWNLOADED_FILES = os.path.join(DATASET_INFO_CSV_DIR, 'downloaded.txt')

# Labels Column Names
MIMIC_LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema",
          "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
          "Lung Opacity","No Finding","Pleural Effusion","Pleural Other",
          "Pneumonia","Pneumothorax","Support Devices"]

# ====== PARAMETERS FOR FEATURE EXTRACTION IN NLP (Graph) ======

NUM_WORKERS = 8 # Number of parallel workers for processing (Threads)

# ======================== Hyperparameters for TRAINING =================================
NUM_EPOCHS = 10
LEARNING_RATE_TRANSFORMER = 1e-5 # For Transformer Blocks
LEARNING_RATE_CLASSIFIER = 1e-4 # For Classifier Head
BATCH_SIZE = 16

UNBLOCKED_LEVELS = 2 # Number of unblocked levels in the Swin Transformer (from end to start)

SWIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models', 'gnn_swin')
SWIN_MODEL_SAVE_PATH = os.path.join(SWIN_MODEL_DIR, 'swin_model.pth')
SWIN_STATS_PATH = os.path.join(SWIN_MODEL_DIR, 'src', 'swin_stats.json')

# Dataset Split
TRAIN_TEST_SPLIT = 0.8 # Ratio of training to testing data
VALIDATION_SPLIT = 0.1 # Ratio of validation to testing data
TEST_SPLIT = 0.1 # Ratio of testing data

# ====== PARAMETERS FOR FEATURE EXTRACTION IN XAI (Attention Map) ======

ATTENTION_MAP_THRESHOLD = 0.5 # Threshold for attention map analysis


# ===== PARAMETERS FOR SIMILARITY CALCULATION ======

# Threshold for similarity calculation between graph nodes and medical reports
SIMILARITY_THRESHOLD = 0.7