import numpy as np
from matplotlib import pyplot as plt
import joblib

import params

CHANNEL_NAMES = ["EEG6", "EMG6", "MOT6"]


def plot(value_seq, score_seq, name,show=True):
    x = [i for i in range(len(value_seq))]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x, value_seq)
    axes[1].scatter(x, score_seq * 1000, c='green')
    if len(value_seq) >= 3 * params.MAX_SEQ_SIZE - 1:
        plt.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
        plt.plot([2 * params.MAX_SEQ_SIZE, 2 * params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
    elif len(value_seq) >= 2 * params.MAX_SEQ_SIZE - 1:
        plt.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
    plt.title(name)
    plt.tight_layout()
    plt.savefig("figs/%s.png" % name)

    if show:
        plt.show()


def plot3c(value_seq, score_seq, name, subtitles, n_channels=3, show=True):
    plt.figure()

    x = [i for i in range(value_seq.shape[-1])]
    fig, axes = plt.subplots(2, n_channels, figsize=(12, 8))
    for i in range(n_channels):
        vs = value_seq[i, :]
        ss = score_seq[i, :]
        axes[0, i].plot(x, vs, [-0.2, 0.2])
        axes[1, i].scatter(x, ss * 1000, c='green')
        axes[1, i].set_ylim(-0.2, 0.2)
        for ax in [axes[0, i], axes[1, i]]:
            if len(vs) >= 3 * params.MAX_SEQ_SIZE - 1:
                ax.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
                ax.plot([2 * params.MAX_SEQ_SIZE, 2 * params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
                ax.text(params.MAX_SEQ_SIZE / 2, 0.15, subtitles[0])
                ax.text(3 * params.MAX_SEQ_SIZE / 2, 0.15, subtitles[1])
                ax.text(5 * params.MAX_SEQ_SIZE / 2, 0.15, subtitles[2])

            elif len(vs) >= 2 * params.MAX_SEQ_SIZE - 1:
                ax.plot([params.MAX_SEQ_SIZE, params.MAX_SEQ_SIZE], [-0.2, .2], c='r')
            ax.set_title(CHANNEL_NAMES[i])
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig("figs/%s.png" % name)
    if show:
        plt.show()

def plot_id(idx, show=False):
    val = np.squeeze(val_seqs[idx])
    label = labels[idx]
    lbw = lbws[idx]
    lbw_names = [idx2lb[jj] for jj in lbw]
    prediction = np.loadtxt("out/predicted.txt")[idx]
    pred_id = np.argmax(prediction)
    # print(label)
    label_id = np.nonzero(label)[0][0]
    print("LB: ", label_id, label, lbw[1], lbw_names)
    shs = shaps[idx][pred_id, :]
    print(shs.shape, val.shape)
    # print(label_id)
    name = "%sX_T_%s_%s_P_%s_%s" % (idx + 1, label_id, idx2lb[label_id], pred_id, idx2lb[pred_id])
    if params.THREE_CHAINS:
        plot3c(val, shs, name, lbw_names, show=show)
    else:
        plot(val, shs, name, show=show)


if __name__ == "__main__":
    val_seqs, labels, lbws, shaps, idx2lb = joblib.load("out/xmodel.pkl")
    shaps = np.squeeze(np.asarray(shaps))
    print(idx2lb)
    # print(len(val_seqs), len(val_seqs[0]), val_seqs[0].shape)
    # exit(-1)
    for i in range(2000):
        plot_id(i, show=False)
    exit(-1)
    while True:
        idx = int(input("Enter Test Index: "))
        if idx == -1:
            print("Exit")
            exit(-1)
        idx = idx - 1
        plot_id(idx, show=False)