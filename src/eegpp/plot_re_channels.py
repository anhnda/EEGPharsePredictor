from matplotlib import pyplot as plt
import numpy as np
import glob


def load_data(root_dir="outx"):
    PATTERN = "%s/M_*" % root_dir

    def get_type(s):
        select = 1
        if s[3] == "F":
            select = 0
        return s[:3], select

    def parse_channel(folder_path):
        parts = folder_path.split("/")[-1].split("_")
        channels = []
        for p in parts[1:4]:
            # print(p)
            channel, select = get_type(p)
            if select:
                channels.append(channel)
        return "+".join(channels)

    d_re = {}
    for folder_path in glob.glob(PATTERN):
        print(folder_path)
        channel = parse_channel(folder_path)
        log_path = folder_path + "/log.txt"
        with open(log_path, "r") as f:
            lines = f.readlines()
            best_re = lines[-1].strip().split(",")
            f1 = float(best_re[2])
            auc = float(best_re[6])
            aupr = float(best_re[7][:-5])
            d_re[channel] = [f1, auc, aupr]
    return d_re


def plot():
    d_re = load_data()
    from utils import ensureDir
    ensureDir("figs")
    res = []
    key_orders = ['EGG', 'EGG+EMG', 'EGG+MOT', 'EGG+EMG+MOT', 'EMG', 'MOT', 'EMG+MOT']
    LABELS = ["F1", "AUC", "AUPR"]
    for key in key_orders:
        values = d_re[key]
        res.append(values)
    ar = np.asarray(res)
    plt.figure()
    for i in range(3):
        y = ar[:, i]
        print("Y", y)
        plt.scatter(np.arange(len(key_orders)), y, label=LABELS[i])
    plt.legend()
    plt.xticks(np.arange(len(key_orders)), key_orders, rotation=45)
    plt.title("Model Performances w.r.t. Channels")
    plt.tight_layout()
    plt.savefig("figs/model_performances.png")
    plt.show()

def plot2():
    d_re = load_data(root_dir="out")
    d_re2 = load_data(root_dir="outx")
    from utils import ensureDir
    ensureDir("figs")
    res = []
    resx = []
    key_orders = ['EGG', 'EGG+EMG', 'EGG+MOT', 'EGG+EMG+MOT', 'EMG', 'MOT', 'EMG+MOT']
    LABELS = ["F1X_3O", "AUC_3O", "AUPR_3O"]
    LABELSX = ["F1X", "AUC", "AUPR"]

    for key in key_orders:
        values = d_re[key]
        res.append(values)
        valuesx = d_re2[key]
        resx.append(valuesx)
    ar = np.asarray(res)
    arx = np.asarray(resx)
    plt.figure()
    for i in range(3):
        y = ar[:, i]
        print("Y", y)
        plt.scatter(np.arange(len(key_orders)), y, label=LABELS[i])
        yx = arx[:, i]
        plt.scatter(np.arange(len(key_orders)), yx, label = LABELSX[i] )

    plt.legend()
    plt.xticks(np.arange(len(key_orders)), key_orders, rotation=45)
    plt.title("Model Performances w.r.t. Channels")
    plt.tight_layout()
    plt.savefig("figs/model_performances_all.png")
    plt.show()

if __name__ == "__main__":
    plot()
