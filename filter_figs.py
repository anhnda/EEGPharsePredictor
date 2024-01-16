import glob
from utils import ensureDir
import shutil

def get_info(path):
    file_name = path.split("/")[-1]
    parts = file_name.split("_")
    true_lb = parts[3]
    pred_lb = parts[6].split(".")[0]
    fid = parts[0]
    return fid, true_lb, pred_lb

def rep_star(s):
    return s.replace("*", "Star")
def filter_figs():
    ensureDir("Fig_StarWrong")
    ensureDir("Fig_NonStarWrong")
    for path in glob.glob("figs/*.png"):
        fid, true_lb, pred_lb = get_info(path)
        if true_lb != pred_lb:
            print(true_lb, pred_lb)
            if true_lb.__contains__("*"):
                shutil.copy(path, "Fig_StarWrong/"+fid+"_"+rep_star(true_lb)+"_"+rep_star(pred_lb)+".png")
            else:
                shutil.copy(path, "Fig_NonStarWrong/"+fid+"_"+rep_star(true_lb)+"_"+rep_star(pred_lb)+".png")



if __name__ == "__main__":
    filter_figs()