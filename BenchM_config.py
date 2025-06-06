
#######################
### Benchmark Model ###
#######################


GLOBAL_SEED = 42


# Model Hyperparameters
NUM_IN_FEAT = 5460
NUM_OUT_CLASSES = 2

MLP_WIDTH_LIST = [5460, 512, 128, 32, 2]


# Training

SPARSITY_GRIDS_INFO = [[1e-5,1e-4],1e-6]
#OPTIM_SPARSITY = #FILL#

CV_COUNT = 10

LEARNING_RATE = 0.001
NUM_WORKERS = 8
BATCH_SIZE = 16
MIN_EPOCHS = 1
VAL_STOP_PATIENCE = 10
MAX_EPOCHS = 10**3
ACCELERATOR = "gpu"
DEVICES = [1]
PRECISION = 32


# Data
DATA_SUPERDIR = "./data/data"
ICN_NII_PATH = "./data/Neuromark_fMRI_2.1_modelorder-multi.nii"
ICN_LABEL_PATH = "./data/Neuromark_fMRI_2.1_modelorder-multi.txt"

DATA_SPLIT = [0.8, 0.2] #split of dev set
SPLIT_SEED = 42

