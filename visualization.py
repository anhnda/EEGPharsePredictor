import numpy as np
from matplotlib import pyplot as plt
import joblib

import params


def plot(value_seq, name):
    fig = plt.figure()
    x = [i for i in range(len(value_seq))]
    plt.plot(x,value_seq)
    if len(value_seq >= 3 * params.MAX_SEQ_SIZE - 1):
        plt.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
        plt.plot([2*params.MAX_SEQ_SIZE, 2*params.MAX_SEQ_SIZE], [-0.2, .2], c='r')

    plt.title(name)
    plt.tight_layout()
    plt.savefig("figs/%s.png"%name)

if __name__ == "__main__":
    val_seqs, labels, idx2lb = joblib.load("out/test_data.pkl")
    print(idx2lb)
    while True:
        idx = int(input("Enter Test Index: "))
        if idx == -1:
            print("Exit")
            exit(-1)
        idx = idx - 1
        val = val_seqs[idx]
        label = labels[idx]
        prediction = np.loadtxt("out/predicted.txt")[idx]
        pred_id = np.argmax(prediction)
        # print(label)
        label_id = np.nonzero(label)[0][0]
        # print(label_id)
        name = "%s_T_%s_%s_P_%s_%s" % (idx+1, label_id, idx2lb[label_id], pred_id, idx2lb[pred_id])
        plot(val, name)
