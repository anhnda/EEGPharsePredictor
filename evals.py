import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import seaborn as sns
from matplotlib import pyplot as plt

LB_DICT = {'W': 0, 'W*': 1, 'NR': 2, 'NR*': 3, 'R': 4, 'R*': 5}
LB_DICT = { v : k for k, v in LB_DICT.items()}
LBS = [LB_DICT[i] for i in range(6)]
def get_confussion_from_list(lbs, preds, n_class):
    cfs_matrix = np.zeros((n_class, n_class))
    for i in range(len(lbs)):
        if lbs[i] >= n_class or preds[i] >= n_class:
            continue
        cfs_matrix[lbs[i], preds[i]] += 1
    print(np.sum(cfs_matrix))
    return cfs_matrix
def plot_cfs_matrix(cfs_matrix, show=False, out_dir=None):
    plt.figure()
    print(cfs_matrix)
    fig, ax = plt.subplots()
    sns.heatmap(cfs_matrix, cmap=sns.color_palette("flare", as_cmap=True), xticklabels=LBS, yticklabels=LBS)
    ax.xaxis.tick_top()
    plt.ylabel("True")
    plt.xlabel("Prediction")
    plt.savefig("%s/confusion_matrix.png" % out_dir)
    if show:
        plt.show()

def create_confusion(y_true, y_pred, n_class):
    cl_true = np.nonzero(y_true)[1]
    cl_pred = np.nonzero(y_pred)[1]
    cfs_matrix = np.zeros((n_class, n_class))
    for i in range(len(y_pred)):
        cfs_matrix[cl_true[i],cl_pred[i]] += 1
    MX = 500
    cfs_matrix[cfs_matrix > MX] = MX
    print(cfs_matrix)
    print(np.sum(cfs_matrix))
    # return np.log(cfs_matrix + 1)
    return cfs_matrix

def eval_multiclasses(y_true, y_score, combine=True):
    if combine:
        y_true, y_score = combine_type(y_true, y_score)
    nr, nc = y_true.shape
    print(np.sum(y_true, axis=0))
    lb = np.argmax(y_score, axis=1)
    print(len(lb))
    tps = [(i, lb[i]) for i in range(len(lb))]
    pred = np.zeros(y_score.shape, dtype=int)
    ids = tuple(np.transpose(tps))

    # print(ids)
    pred[ids] = 1
    print(np.sum(pred, axis=0))
    print(pred[-1,:])
    # print(pred)
    # print(pred)
    cfs_matrix = create_confusion(y_true, pred, n_class=6)
    plot_cfs_matrix(cfs_matrix, show=True)
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
    run(combine=False)
    # ar = np.random.random((2,4))
    # print(ar[:, 2:4])
    # {0: 'W*', 1: 'W', 2: 'NR', 3: 'NR*', 4: 'R*', 5: 'R', 6: 'S2*'}
