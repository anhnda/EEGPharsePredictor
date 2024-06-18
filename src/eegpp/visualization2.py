import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import joblib

import params
import utils
from train import get_model_dirname, parse_x

CHANNEL_NAMES = ["EEG6", "EMG6", "MOT6"]

sm = torch.nn.Softmax(dim=-1)


def plot(value_seq, score_seq, name, show=True, out_dir=None):
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
    plt.savefig("%s/%s.png" % (out_dir, name))

    if show:
        plt.show()


def plot3c(value_seq, score_seq, name, subtitles, n_channels=3, show=True, out_dir="figs"):
    plt.figure()

    x = [i for i in range(value_seq.shape[-1])]
    fig, axes = plt.subplots(2, n_channels, figsize=(12, 8))
    for i in range(n_channels):
        vs = value_seq[i, :]
        ss = score_seq[i, :]
        axes[0, i].plot(x, vs, [-0.2, 0.2])
        axes[1, i].scatter(x, ss * 10, c='green')
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
    plt.savefig("%s/%s.png" % (out_dir, name))
    # if show:
    #     plt.show()


def plot_id(idx, show=False, out_dir=None):
    val = np.squeeze(val_seqs[idx])
    # val = val / np.max(np.abs(val)) * 0.2
    label = labels[idx]
    lbw = lbws[idx]
    epoch_id = epochess[idx]
    lbw_names = []
    for jj in lbw:
        if jj != -1:
            lbw_names.append(idx2lb[jj])
        else:
            lbw_names.append("PAD")




    prediction = preds[idx]
    pred_id = np.argmax(prediction)


    # print(label)
    label_id = np.nonzero(label)[0][0]
    print("LB: ", label_id, label, lbw[1], lbw_names)
    shs = shaps[idx]
    if params.OUT_3C:
        print(shs.shape)
        NC = 3
        if params.TWO_CHAINS:
            NC = 2
        shs = shs.reshape(params.NUM_CLASSES, NC, 3, shs.shape[-1])[:, 1, :, :][pred_id, :]
    else:
        shs = shs[pred_id, :]
    # print("Shape v x: ", shs.shape)
    shs = shs / np.max(np.abs(shs)) * 0.05
    # print()
    print(shs.shape, val.shape, np.max(shs))
    # print(label_id)
    name = "%sX_O_%s_T_%s_%s_P_%s_%s" % (idx + 1, epoch_id, label_id, idx2lb[label_id], pred_id, idx2lb[pred_id])
    if params.THREE_CHAINS or params.TWO_CHAINS:
        nc = 3
        if params.TWO_CHAINS:
            nc = 2
        plot3c(val, shs, name, lbw_names, n_channels=nc, show=show, out_dir=out_dir)
    else:
        plot(val, shs, name, show=show)

    return label_id, pred_id


if __name__ == "__main__":
    parse_x()
    fig_dir = get_model_dirname() + "/figs/" + "%s" % params.TRAIN_ID + "_" + "%s" % params.TEST_ID
    os.system("rm -rf %s/*" % fig_dir)
    utils.ensureDir(fig_dir)

    MODEL_ID = params.TRAIN_ID
    TEST_ID = params.TEST_ID
    model_xpath = "%s/xmodel_%s_%s.pkl" % (get_model_dirname(), MODEL_ID, TEST_ID)
    print("Model xpath: ", model_xpath)
    val_seqs, labels, lbws, shaps, idx2lb, epochess, preds, preds_3os = joblib.load(model_xpath)
    shaps = np.squeeze(np.asarray(shaps))
    epochess = np.squeeze(np.asarray(epochess))
    print(idx2lb)
    # print(len(val_seqs), len(val_seqs[0]), val_seqs[0].shape)
    # exit(-1)
    all_preds = []
    all_lbs = []

    for i in range(1999):
        if i == 20001 or i == len(preds):
            break
        lb, pred = plot_id(i, show=False, out_dir=fig_dir)
        all_preds.append(pred)
        all_lbs.append(lb)
    print(all_lbs)
    print(all_preds)
    from evals import get_confussion_from_list, plot_cfs_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score

    cfs_matrix = get_confussion_from_list(all_lbs, all_preds, 6)
    plot_cfs_matrix(cfs_matrix, False, out_dir=fig_dir)
    mm = 'macro'
    pre = precision_score(all_lbs, all_preds, average=mm)
    rec = recall_score(all_lbs, all_preds, average=mm)
    f1 = f1_score(all_lbs, all_preds, average=mm)
    print("Precision: ", pre)
    print("Recall: ", rec)
    print("F1: ", f1)
    fout = open(fig_dir + "/re_test.txt", "w")
    fout.write("Precision,Recall,F1,%s,%.4f,%.4f,%.4f" % (mm, pre, rec, f1))
    fout.close()
    # exit(-1)
    # while True:
    #     idx = int(input("Enter Test Index: "))
    #     if idx == -1:
    #         print("Exit")
    #         exit(-1)
    #     idx = idx - 1
    #     plot_id(idx, show=False)
