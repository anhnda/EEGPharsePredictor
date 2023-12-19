import numpy as np
from matplotlib import pyplot as plt
import joblib

import params


def plot(value_seq, score_seq, name):
    fig = plt.figure()
    x = [i for i in range(len(value_seq))]
    _, axes = plt.subplots(2,1,figsize = (6,8) )
    axes[0].plot(x,value_seq)
    axes[1].scatter(x, score_seq * 1000, c='green')
    if len(value_seq) >= 3 * params.MAX_SEQ_SIZE - 1:
        plt.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
        plt.plot([2*params.MAX_SEQ_SIZE, 2*params.MAX_SEQ_SIZE], [-0.2, .2], c='r')

    plt.title(name)
    plt.tight_layout()
    plt.show()
    plt.savefig("figs/%s.png"%name)

if __name__ == "__main__":
    val_seqs, labels, shaps, idx2lb = joblib.load("out/xmodel.pkl")
    shaps = np.squeeze(np.asarray(shaps))
    print(idx2lb)
    while True:
        idx = int(input("Enter Test Index: "))
        if idx == -1:
            print("Exit")
            exit(-1)
        idx = idx - 1
        val = val_seqs[idx].reshape(-1)
        label = labels[idx]
        prediction = np.loadtxt("out/predicted.txt")[idx]
        pred_id = np.argmax(prediction)
        # print(label)
        label_id = np.nonzero(label)[0][0]
        shs = shaps[idx][pred_id, :]
        print(shs.shape, val.shape)
        # print(label_id)
        name = "%sX_T_%s_%s_P_%s_%s" % (idx+1, label_id, idx2lb[label_id], pred_id, idx2lb[pred_id])
        plot(val, shs, name)
