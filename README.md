# EEG Phrase Predictor

## Setup
- Requirements: python==3.9
- Installing: 
    ```
        pip install eegpp
    ```
## Configuration
YML_CONFIG_FILE (e.g. data_config_infer.yml) with the following contents

```
datasets:
  time_step: 4000 #milliseconds
  data_dir: "FULL_PATH_TO_THE_FOLDER_WITH_RAW_SIGNAL_FILES"
  tmp_dir : "FULL_PATH_TO_THE_TMP_FOLDER"
  out_dir: "FULL_PATH_TO_THE_OUTPUT_FOLDER"
  seq_files: ["name_of_raw_file_1", "name_of_raw_file_2",...,"name_of_raw_file_n]
  out_seperator: "\t"

```

Example:
```
datasets:
  time_step: 4000 #milliseconds
  data_dir: "/home/EEGData/EEG_test_files"
  tmp_dir : "/home/EEGData/tmp"
  out_dir: "/home/EEGData/EEG_test_files"
  seq_files: ["raw_K3_EEG3_11h.txt", "raw_RS2_EEG1_23 hr.txt", "raw_S1_EEG1_23 hr.txt"]
  out_seperator: "\t" for tab or "," for commas

```

## Running

Command:

```
    python -m eegpp -p PATH_TO_THE_YML_CONFIG_FILE -e -t 0.55 {-n for norule}
```
e.g.

```
    python -m eegpp -p data_config_infer.yml -e -t 0.55
```

No rule:
```
    python -m eegpp -p data_config_infer.yml -e -t 0 -n
```

Add option -l to enter silence mode (no screen output).

For visualization:

First run inference with option -s:

```
    python -m eegpp -p data_config_infer.yml -e -t 0 -n -s
```

Then:

```
    python -m eegpp -p data_config_infer.yml -v
```

To enter visualization mode for the first dataset in data_config_infer.yml. 
EpochID starts with 1

or add -i option to set epoch ids

```
    python -m eegpp -p data_config_infer.yml -v -i EpochID1,EpochId2,...,EpochIDn
```
To plot the signal and prediction of the EpochID (Indexing from 1), separating by commas
