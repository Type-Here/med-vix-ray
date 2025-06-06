import os, json

service_account_var = os.environ.get('SERVICE_ACCOUNT_TOKEN')
if service_account_var and service_account_var[0] == "'":  # If the first character is a single quote, remove it
    service_account_var = service_account_var[1:]
    service_account_var = service_account_var[:-1]  # Remove the last character (single quote)
service_account = json.loads(service_account_var)
print(f"---- Service Account Domain: {service_account['universe_domain']}")

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
NER_GROUND_TRUTH = os.path.join(RADLEX_GRAPH_DIR, 'report_keywords', 'all_comparison_results.json')

# Model Save directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models')

# Report data
MIMIC_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic')

# Set of the available images in all directories file in pickle format
IMAGES_SET_PATHS_AVAILABLE = os.path.join(MIMIC_REPORT_DIR, 'images_set_available.pkl')

# ==== ENVIRONMENT VARIABLES ====

# Dataset path
DATASET_PATH = os.environ.get('MIMIC_DATASET_PATH') # Till "files"-named directory
# FUSE mounted bucket path; (e.g. bucket_name; gs:// will be added in the code)
BUCKET_PREFIX_PATH = os.environ.get('BUCKET_PREFIX_PATH') # Till "files"-named directory
# GCP billing project name if Bucket requires billing to requester. If
try:
    BILLING_PROJECT = os.environ.get('BILLING_PROJECT') # GCP project name
except (AttributeError, KeyError):
    print("[WARNING] GCP billing project not set. BILLING_PROJECT will be None.")
    BILLING_PROJECT = None # If not set, it will be None

SERVICE_ACCOUNT_TOKEN =  None # Service account token for GCP
# os.environ.get('SERVICE_ACCOUNT_TOKEN', None)
# SERVICE_ACCOUNT_TOKEN = service_account # Service account token for GCP if using json token

# ================================

# Directory where csv files are stored, containing the dataset information
DATASET_INFO_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic', 'info')

# Directory where the dataset is split into train, validation, and test sets manually
SPLIT_DATASET_DIR = os.path.join(DATASET_INFO_CSV_DIR, 'split_dataset')
# Directory where the dataset is split into train, validation, and test sets
# from the original dataset platform (PhysioNet)
MIMIC_SPLIT_DIR = os.path.join(DATASET_INFO_CSV_DIR, 'mimic_split')
CSV_METADATA_DIR = os.path.join(DATASET_INFO_CSV_DIR, 'csv_metadata')
PICKLE_METADATA_DIR = os.path.join(DATASET_INFO_CSV_DIR, 'pickle_metadata')

# List of Downloaded Files (if partial_list is not empty) .txt file
DOWNLOADED_FILES = os.path.join(DATASET_INFO_CSV_DIR, 'downloaded.txt')

# MIMIC Split CSV Info
MIMIC_SPLIT_CSV = os.path.join(DATASET_INFO_CSV_DIR, 'mimic-cxr-split.csv')

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
LEARNING_RATE_INPUT_LAYER = 1e-6 # For Input Layer
# If the learning rate is too high, the model may not converge or may diverge.

BATCH_SIZE = 16

# Early Stopping
EARLY_STOPPING_PATIENCE = NUM_EPOCHS // 3 + 1 # Number of epochs with no improvement after which training will be stopped
# For 10 epochs, 4 epochs of patience; For 5 epochs, 2 epochs of patience

UNBLOCKED_LEVELS = 3 # Number of unblocked levels in the Swin Transformer (from end to start)

SWIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models', 'finetuned')
SWIN_MODEL_SAVE_PATH = os.path.join(SWIN_MODEL_DIR, 'swin_model.pth')
SWIN_STATS_PATH = os.path.join(SWIN_MODEL_DIR, 'src', 'swin_stats.json')

# Dataset Split
TRAIN_TEST_SPLIT = 0.8 # Ratio of training to testing data
VALIDATION_SPLIT = 0.1 # Ratio of validation to testing data
TEST_SPLIT = 0.1 # Ratio of testing data

# ====== PARAMETERS FOR GRAPH WEIGHTS TRAINING ======
EPOCH_GRAPH_INTEGRATION = NUM_EPOCHS // 5 # Number of epochs for graph integration (Default: 20%)
INJECT_BIAS_FROM_THIS_LAYER = 2 # Number of which layer to start injecting bias from

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
ALPHA_GRAPH = 0.2 # Alpha correction factor for the graph attention mechanism Initial value (Learnable)

# LOSS FROM GRAPH
LAMBDA_SIM=0.6 # Regularization parameter for the loss function # Similarity loss
LAMBDA_KL=0.2 # KL divergence parameter for the loss function

# ETA FOR GRAPH NUDGER MODULE LEARNING RATE
ETA_GRAPH = 0.1 # Learning rate for the graph nudger module

# ====== PARAMETERS FOR FEATURE EXTRACTION IN XAI (Attention Map) ======

# Threshold for attention map analysis on active regions identification
# Values:
# Float between 0.0 and 1.0: Use static threshold for attention map analysis
# String "adaptive": Use adaptive otsu-like threshold for attention map analysis
# String "percentile": Use percentile threshold for attention map analysis
# (percentile on minimum and maximum values; curricular on training epochs)
ATTENTION_MAP_THRESHOLD = 'percentile' # Threshold for attention map analysis

MAX_REGIONS_PER_IMAGE = 5 # Maximum number of regions to be extracted from the attention map for each image

# ===== PARAMETERS FOR SIMILARITY CALCULATION ======

# Threshold for similarity calculation between graph nodes and medical reports
SIMILARITY_THRESHOLD = 0.7