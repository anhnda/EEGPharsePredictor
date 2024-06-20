import numpy as np


def correct_star(prediction_scores, threshold):
    predicted_lbids = np.argmax(prediction_scores, axis=-1)

    ids = np.arange(0, prediction_scores.shape[0])
    ii = tuple(np.asarray(list(zip(ids, predicted_lbids))).T)
    max_ids_cor = ii
    mx_scores = prediction_scores[max_ids_cor]
    prediction_scores[max_ids_cor] = 0
    second_ids = np.argmax(prediction_scores, axis=-1)
    ids_to_correct = np.argwhere(mx_scores <= threshold).reshape(-1)
    corrected_lb = []
    for i in ids_to_correct:
        second_id = second_ids[i]
        lb0 = predicted_lbids[i]
        if lb0 % 2 == 0:
            if second_id % 2 == 1:
                corrected_lb.append(second_id)
            else:
                corrected_lb.append(lb0 + 1)
        else:
            corrected_lb.append(lb0)
    predicted_lbids[ids_to_correct] = corrected_lb
    return predicted_lbids


def correct_4wr(predictions_lbids, ns=4):
    ids = np.argwhere(predictions_lbids >= 4).reshape(-1)
    for i in ids:
        if i >= ns:
            mx = max(predictions_lbids[i - ns:i])
            if mx <= 1:

                if mx == 0:
                    predictions_lbids[i] = 0
                else:
                    predictions_lbids[i] = 1


if __name__ == '__main__':
    ss = np.asarray([0,0,0,4,4,4])
    correct_4wr(ss, ns=3)
    print(ss)
