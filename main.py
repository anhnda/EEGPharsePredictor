from src.eegpp.inference import infer_cmd
from src.eegpp.train_all_data import run_training_all
from src.eegpp import params
if params.DATA_CONFIG_PATH.__contains__("train"):
    run_training_all()
else:
    infer_cmd()
