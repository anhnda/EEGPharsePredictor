import os
from pathlib import Path
DATA_DIR = os.path.abspath(os.path.dirname(__file__))
S_DATA_DIR = "/home/DATA/EEG/TRAIN_2025_09_DXM/raw text files/"
S_LABEL_DIR = "/home/DATA/EEG/TRAIN_2025_09_DXM/cFFT/"
#DATA_DIR = "/home/DATA/EEG/WDIR/"
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
DUMP_DATA_DIR = os.path.join(DATA_DIR, 'dump')

os.makedirs(DUMP_DATA_DIR, exist_ok=True)

SEQ_FILES = [
    str(Path(S_DATA_DIR, "ELES 1A_EEG3.txt")),
    str(Path(S_DATA_DIR, "ELES 2B_EEG1.txt")),
    str(Path(S_DATA_DIR, "ELES 3A_EEG6.txt")),
    str(Path(S_DATA_DIR, "ELES 4A_EEG4.txt")),
    str(Path(S_DATA_DIR, "ELES 1A_EEG5.txt")),
    str(Path(S_DATA_DIR, "ELES 2B_EEG4.txt")),
    str(Path(S_DATA_DIR, "ELES 3B_EEG3.txt")),
    str(Path(S_DATA_DIR, "POT B_EEG7.txt")),
    str(Path(S_DATA_DIR, "ELES 2A_EEG1.txt")),
    str(Path(S_DATA_DIR, "ELES 2B_EEG5.txt")),
    str(Path(S_DATA_DIR, "ELES 3B_EEG6.txt"))
    #str(Path(RAW_DATA_DIR, "raw_K3_EEG3_11h.txt")),
    #str(Path(RAW_DATA_DIR, "raw_RS2_EEG1_23 hr.txt")),
    #str(Path(RAW_DATA_DIR, "raw_S1_EEG1_23 hr.txt")),
    #str(Path(RAW_DATA_DIR, "K1_EEG1_SAL.csv")),
    #str(Path(RAW_DATA_DIR, "K1_EEG7_SAL.csv")),
    #str(Path(RAW_DATA_DIR, "K2_EEG4_SAL.csv")),
    #str(Path(RAW_DATA_DIR, "K2_EEG5_SAL.csv")),
    #str(Path(RAW_DATA_DIR, "K4_EEG7_SAL.csv")),
]

LABEL_FILES = [
    str(Path(S_LABEL_DIR, "ELES 1A_EEG3_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 2B_EEG1_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 3A_EEG6_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 4A_EEG4_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 1A_EEG5_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 2B_EEG4_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 3B_EEG3_cFFT.txt")),
    str(Path(S_LABEL_DIR, "POT B EEG7_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 2A_EEG1_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 2B_EEG5_cFFT.txt")),
    str(Path(S_LABEL_DIR, "ELES 3B_EEG6_cFFT.txt"))
    #str(Path(RAW_DATA_DIR, "K3_EEG3_11h.txt")),
    #str(Path(RAW_DATA_DIR, "RS2_EEG1_23 hr.txt")),
    #str(Path(RAW_DATA_DIR, "S1_EEG1_23 hr.txt")),
    #str(Path(RAW_DATA_DIR, "K1_EEG1_11h.txt")),
    #str(Path(RAW_DATA_DIR, "K1_EEG7_11h.txt")),
    #str(Path(RAW_DATA_DIR, "K2_EEG4_11h.txt")),
    #str(Path(RAW_DATA_DIR, "K2_EEG5_11h.txt")),
    #str(Path(RAW_DATA_DIR, "K4_EEG7_11h.txt")),
]

DUMP_DATA_FILES = {
    "train": [
        str(Path(DUMP_DATA_DIR, f"dump_eeg_{i+1}.pkl")) for i in range(len(SEQ_FILES))
    ],
    "infer": [
        str(Path(DUMP_DATA_DIR, f"dump_eeg_{i+1}_infer.pkl")) for i in range(len(SEQ_FILES))
    ],
}
