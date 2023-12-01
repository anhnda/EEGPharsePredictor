import numpy as np
from matplotlib import pyplot as plt
import joblib
def plot(value_seq, name):
    fig = plt.figure()
    x = [i for i in range(len(value_seq))]
    plt.plot(x,value_seq)
    plt.title(name)
    plt.tight_layout()
    plt.savefig("figs/%s.png"%name)

if __name__ == "__main__":
    val_seqs, labels, idx2lb = joblib.load("out/test_data.pkl")
    print(idx2lb)
    while True:
        print("Enter Test Index: ")
        idx = int(input())
        if idx == -1:
            print("Exit")
            exit(-1)
        val = val_seqs[idx]
        label = labels[idx]
        print(label)
        label_id = np.nonzero(label)[0]
        print(label_id)
        name = "%s_%s_%s" % (idx, label_id, idx2lb[label_id])
        plot(val, name)
