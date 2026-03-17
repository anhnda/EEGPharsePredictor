# EEG Phase Predictor ver 2.3

**Note: This is beta version, use for training with default dataset and inference only
** Support 128Hz and 256Hz input sampling **

## Setup

- Requirements: python >= 3.10
- Installing:

```aiignore
  pip install eegpp3-beta
```

## Train with default dataset

```aiignore
    python -m eegpp3 --mode "train" --model_type <mode_type> --lr <learning_rate> --batch_size <batch_size> --n_epochs <num_epochs> --n_splits <num_folds> --resume_checkpoint <resume_from_checkpoint>
```

ex: `python -m eegpp3 --mode "train" --model_type "stftcnn1dnc" --n_epochs 20 --n_splits 10 --resume_checkpoint False`

## Inference

1. Old version (**Recommend**)

```aiignore
    python -m eegpp3 -p <path_to_infer_config.yml>
```

ex: `python -m eegpp3 -p "data_config_infer.yml"`

An example of the configuration file:
```
datasets:
  time_step: 4000 #milliseconds
  data_dir: "/home/user/data"
  tmp_dir: "/home/user/tmp"
  out_dir: "/home/user/out"
  seq_files: ["raw_K3_EEG3_11h.txt", "raw_RS2_EEG1_23 hr.txt"]
  template_files: [] # ["K3_EEG3_11h.txt", "RS2_EEG1_23 hr.txt", "S1_EEG1_23 hr.txt"] # if no template, set to : [ ] or remove this line
  out_seperator: "\t"
```
Change directory (dir) with the correct path
2. New version

```aiignore
  python -m eegpp3 --mode "infer" --data_path <path_to_data_file> --infer_path <path_to_saving_file> --model_type <model_type>
```

ex:
`python -m eegpp3 --mode "infer" --data_path "./dump_eeg_1.pkl" --infer_path './inference_result.txt" --model_type "stftcnn1dnc"`

## Model type

- stftcnn1dnc: Multi-channels STFT-CNN
