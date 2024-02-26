import glob
import os

from utils import ensureDir
import shutil
from train import get_model_dirname, parse_x
import params
def get_info(path):
    file_name = path.split("/")[-1]
    parts = file_name.split("_")
    true_lb = parts[5]
    pred_lb = parts[8].split(".")[0]
    fid = parts[0]
    eid = parts[2]
    return fid, eid, true_lb, pred_lb

def rep_star(s):
    return s.replace("*", "Star")
def filter_figs():
    fig_dir = get_model_dirname() + "/figs/" + params.TRAIN_ID + "_" + params.TEST_ID

    fig_dir1 = get_model_dirname() + "/Fig_StarWrong/" + params.TRAIN_ID + "_" + params.TEST_ID
    fig_dir2 = get_model_dirname() + "/Fig_NonStarWrong/" + params.TRAIN_ID + "_" + params.TEST_ID

    os.system("rm -rf " + fig_dir1)
    os.system("rm -rf " + fig_dir2)
    ensureDir(fig_dir1)
    ensureDir(fig_dir2)
    ns = 0

    for path in glob.glob("%s/*.png" % fig_dir):
        if not path.__contains__("X_O"):
            continue

        fid, eid, true_lb, pred_lb = get_info(path)
        if true_lb.__contains__("*"):
            ns += 1
        if true_lb != pred_lb:
            print(true_lb, pred_lb)
            if true_lb.__contains__("*"):
                shutil.copy(path, fig_dir1 + "/"+eid+"_"+rep_star(true_lb)+"_"+rep_star(pred_lb)+".png")
            else:
                shutil.copy(path, fig_dir2 +  "/"+eid+"_"+rep_star(true_lb)+"_"+rep_star(pred_lb)+".png")
    print(ns)


if __name__ == "__main__":
    parse_x()
    filter_figs()