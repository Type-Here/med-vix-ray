import os
# Radlex Ontology
RADLEX_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology', 'data')
RADLEX_DATA = os.path.join(RADLEX_DATA_DIR, 'RadLex.owl')

RADLEX_GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ontology')
RADLEX_GRAPH = os.path.join(RADLEX_GRAPH_DIR, 'radlex_graph.json')

FILTER_RADLEX_JSON = os.path.join(RADLEX_DATA_DIR, 'filter.json')

# Manual Graph Data
MANUAL_GRAPH_DIR = os.path.join(RADLEX_GRAPH_DIR, "data")
MANUAL_GRAPH = os.path.join(MANUAL_GRAPH_DIR, 'graph_gnn.json')

# NER Comparison Keywords with graph
NER_GROUND_TRUTH = os.path.join(RADLEX_GRAPH_DIR, 'reports_keywords', 'comparison_results_old.json')

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

MIMIC_LABELS_MAP_TO_GRAPH_IDS = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 13,
    "Pleural Effusion": 8,
    "Pleural Other": 9,
    "Pneumonia": 10,
    "Pneumothorax": 11,
    "Support Devices": 12
}

# ====== PARAMETERS FOR FEATURE EXTRACTION IN NLP (Graph) ======

NUM_WORKERS = 8 # Number of parallel workers for processing (Threads)

# ========================================= TRAINING PARAMETERS ===============================================
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

# ====== PARAMETERS FOR GRAPH WEIGHTS TRAINING ======
EPOCH_GRAPH_INTEGRATION = NUM_EPOCHS // 5 # Number of epochs for graph integration (Default: 20%)

# For the available Labels Correlation Edges in graph: if correlation edge is present and: (label = pathology)
# - the pathologies are both present in a specific image (1.0 both) update the weight of +positive_weight_corr;
# - if one of the labels is present and the other is not (1.0, 0.0) update the weight of -negative_weight_corr;
# Default privilege for positive samples
POSITIVE_WEIGHT_CORR = 0.7 # Weight for positive samples
NEGATIVE_WEIGHT_CORR = -0.1 # Weight for negative samples

# For Findings Edges: if a pathology is linked to a finding in the report add +positive_weight_finding;
# if a pathology is not linked to a finding in the report add -negative_weight_finding
# Default privilege for positive samples since the findings are statistically linked to the pathologies
POSITIVE_WEIGHT_FINDING = 0.7 # Weight for positive samples
NEGATIVE_WEIGHT_FINDING = -0.2 # Weight for negative samples

# ALPHA CORRECTION IN ATTENTION BY THE GRAPH
ALPHA_GRAPH = 0.2 # Alpha correction factor for the graph attention mechanism

# LOSS FROM GRAPH
LAMBDA_REG=0.05 # Regularization parameter for the loss function

# ETA FOR GRAPH NUDGER MODULE LEARNING RATE
ETA_GRAPH = 0.1 # Learning rate for the graph nudger module

# ====== PARAMETERS FOR FEATURE EXTRACTION IN XAI (Attention Map) ======

ATTENTION_MAP_THRESHOLD = 0.5 # Threshold for attention map analysis


# ===== PARAMETERS FOR SIMILARITY CALCULATION ======

# Threshold for similarity calculation between graph nodes and medical reports
SIMILARITY_THRESHOLD = 0.7