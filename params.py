DID = 1
DATA_DIR = "./EEG_test_files-2"
SEQUENCE_FILE = "%s/ELES_K3_EEG3_SAL_11h.txt" % DATA_DIR
LABEL_FILE = "%s/ELES_20211002_K3_EEG3_10775-fenyvaltasig-PN_OK másolata_cFFT_11h_P.txt" % DATA_DIR
DUMP_FILE = "%s/dump_egg_%s.pkl" % (DATA_DIR, DID)
DUMP_FILE_PATTERN = "%s/dump_egg_" % DATA_DIR + "%s.pkl"
#
# DID = 2
# DATA_DIR = "./EEG_test_files-2"
# SEQUENCE_FILE = "%s/raw_S1_EEG1_23 hr.txt" % DATA_DIR
# LABEL_FILE = "%s/S1_EEG1_23 hr.txt" % DATA_DIR
# DUMP_FILE = "%s/dump_egg_%s.pkl" % (DATA_DIR, DID)

#
# DID = 3
# DATA_DIR = "./EEG_test_files-2"
# SEQUENCE_FILE = "%s/raw_RS2_EEG1_23 hr.txt" % DATA_DIR
# LABEL_FILE = "%s/RS2_EEG1_23 hr.txt" % DATA_DIR
# DUMP_FILE = "%s/dump_egg_%s.pkl" % (DATA_DIR, DID)
W_DIR = "."
NUM_CLASSES = 7


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
TWO_SIDE = 3
MID = 1
DEVICE = "mps"
MODE_TYPE = "CNN2C"
SIDE_FLAG = TWO_SIDE
CRITERIA = "F1X"
OFF_EGG = False
OFF_EMG = False
OFF_MOT = True
OUT_3C = True

TRAIN_ID = 1
TEST_ID = 1
