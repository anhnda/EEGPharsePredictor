import os
DATA_CONFIG_PATH = "data_config_train.yml"
DATA_DIR = None
DUMP_FILE_PATTERN = None
W_DIR = "."
NUM_CLASSES = 7
DID = 1


def get_dump_filename():

    return "%s/dump_egg_%s.pkl" % (DATA_DIR, DID)


MAX_SEQ_SIZE = 1024
D_MODEL = 64
RD_SEED = 1
BATCH_SIZE = 10
N_EPOCH = 20
THREE_CHAINS = False
TWO_CHAINS = True

assert (TWO_CHAINS or THREE_CHAINS) and (TWO_CHAINS != THREE_CHAINS)
LEFT = 2
MID = 1
TWO_SIDE = 5 # = WINDOWS_SIZE
WINDOW_SIZE = TWO_SIDE
POS_ID = int(WINDOW_SIZE / 2)
DEVICE = None
MODE_TYPE = "CNN2C"
SIDE_FLAG = TWO_SIDE
CRITERIA = "F1X"
OFF_EGG = False
OFF_EMG = False
OFF_MOT = True
OUT_3C = True

C_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "%s/model_1.pkl" % C_DIR
TRAIN_ID = 1
TEST_ID = 1

STAR_THRESHOLD = 0.55
RULE = True
W_STAR = 1
