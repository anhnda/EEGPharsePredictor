import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def eval_multiclasses(y_true, y_score, combine=True):
    if combine:
        y_true, y_score = combine_type(y_true, y_score)
    nr, nc = y_true.shape
    print(np.sum(y_true, axis=0))
    lb = np.argmax(y_score, axis=1)
    tps = [(i, lb[i]) for i in range(len(lb))]
    pred = np.zeros(y_score.shape, dtype=int)
    pred[tuple(np.transpose(tps))] = 1
    pred[lb] = 1
    # print(pred)

    pres, recs, f1s = [], [], []
    for i in range(nc):
        y_i = y_true[:, i]
        s_i = pred[:, i]
        pres.append(precision_score(y_i, s_i))
        recs.append(recall_score(y_i, s_i))
        f1s.append(f1_score(y_i, s_i))
    print(pres)
    print(recs)
    print(f1s)
    return pres, recs, f1s


def combine_type(y_true, y_score):
    yy = []
    ys = []
    for i in range(3):
        yi = y_true[:, 2 * i:2 * (i + 1)]
        si = y_score[:, 2 * i:2 * (i + 1)]
        yi = np.sum(yi, axis=1, keepdims=True)
        si = np.sum(si, axis=1, keepdims=True)
        yy.append(yi)
        ys.append(si)
    y_true = np.concatenate(yy, axis=1)
    y_score = np.concatenate(ys, axis=1)
    return y_true, y_score


def run(combine=True):
    y_true = np.loadtxt("out/true.txt", dtype=int)
    y_score = np.loadtxt("out/predicted.txt")
    # print(y_true, y_score)
    eval_multiclasses(y_true, y_score, combine=combine)


if __name__ == "__main__":
    run(combine=True)
    # ar = np.random.random((2,4))
    # print(ar[:, 2:4])
