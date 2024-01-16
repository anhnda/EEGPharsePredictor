# DATA_DIR = "./EEG_test_files"
# SEQUENCE_FILE = "%s/ELES_K3_EEG3_SAL_11h.txt" % DATA_DIR
# LABEL_FILE = "%s/ELES_20211002_K3_EEG3_10775-fenyvaltasig-PN_OK másolata_cFFT_11h_P.txt" % DATA_DIR
# DUMP_FILE = "%s/dump_egg.pkl" % DATA_DIR


DID = 2
DATA_DIR = "./EEG_test_files-2"
SEQUENCE_FILE = "%s/raw_S1_EEG1_23 hr.txt" % DATA_DIR
LABEL_FILE = "%s/S1_EEG1_23 hr.txt" % DATA_DIR
DUMP_FILE = "%s/dump_egg_%s.pkl" % (DATA_DIR, DID)


MAX_SEQ_SIZE = 1024
D_MODEL = 64
RD_SEED = 1
BATCH_SIZE = 10
N_EPOCH = 100
THREE_CHAINS = True
LEFT = 2
TWO_SIDE = 3
MID = 1
DEVICE = "mps"
MODE_TYPE = "CNN3C"
SIDE_FLAG = TWO_SIDE
CRITERIA = "F1X"
