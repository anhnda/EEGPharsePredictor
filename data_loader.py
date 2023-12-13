import params
import utils
import joblib
import os

LABEL_MARKER = "EpochNo"
SEQ_MARKER = "Time"


def load_labels(inp=params.LABEL_FILE):
    fin = open(inp, errors='ignore')

    labels = []
    times = []
    lb_dict = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith(LABEL_MARKER):
            break
    if line == "":
        return [], []

    while True:
        line = fin.readline()
        if line == "":
            break
        if line == "\n":
            continue
        parts = line.split("\t")
        # print(parts)
        epoch_id = parts[0]
        lb_text = parts[1]
        time_text = parts[2]
        label_v = utils.get_insert_dict_index(lb_dict, lb_text)
        time_v = utils.convert_time(time_text)
        labels.append(label_v)
        times.append(time_v)
    fin.close()
    return labels, times, lb_dict


def load_seq_data(times, labels, inp=params.SEQUENCE_FILE):
    fin = open(inp)
    print(times[:10])
    print(labels[:10])
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith(SEQ_MARKER):
            break

    cid = 0
    value_seqs = [[], [], []]
    label_seqs = []
    time_v = -1
    is_exit = False
    mx = -10000
    while not is_exit:
        while time_v < times[cid]:
            line = fin.readline()
            if line == "":
                is_exit = True
                break
            parts = line.split("\t")
            time_text = parts[0]

            value_texts = parts[1:4]
            time_v = utils.convert_time(time_text)
            continue
        cid += 1
        if len(labels) <= cid:
            break
        label_seqs.append(labels[cid])
        c_seqs = [[], [], []]
        is_next_seg = False
        while not is_exit and not is_next_seg:
            assert len(value_texts) == 3
            for i, value_text in enumerate(value_texts):
                v = float(value_text)
                if abs(v) > mx:
                    mx = abs(v)

                c_seqs[i].append(v)
                # print(i, len(c_seqs[i]))
                if i == 0:
                    line = fin.readline()
                    if line == "":
                        is_exit = True
                        break
                    parts = line.split("\t")
                    time_text = parts[0]
                    value_texts = parts[1:4]
                    time_v = utils.convert_time(time_text)
                if time_v >= times[cid]:
                    value_seqs[i].append(c_seqs[i][:params.MAX_SEQ_SIZE])
                    is_next_seg = True

    fin.close()
    return value_seqs, label_seqs, mx


def load_data(force_reload=False):
    if os.path.exists(params.DUMP_FILE) and force_reload is False:
        value_seqs, label_seqs, mx, lb_dict = joblib.load(params.DUMP_FILE)
    else:
        labels, times, lb_dict = load_labels(params.LABEL_FILE)
        value_seqs, label_seqs, mx = load_seq_data(times, labels, params.SEQUENCE_FILE)
        joblib.dump((value_seqs, label_seqs, mx, lb_dict), params.DUMP_FILE)
    print(len(label_seqs), len(value_seqs), mx)
    for i in range(len(label_seqs)):
        # print(len(value_seqs[0]), type(value_seqs[0]))

        assert len(value_seqs[0][i]) == params.MAX_SEQ_SIZE
        # print(len(value_seqs[1]), type(value_seqs[1]))
        assert len(value_seqs[1][i]) == params.MAX_SEQ_SIZE
        assert len(value_seqs[2][i]) == params.MAX_SEQ_SIZE


if __name__ == "__main__":
    load_data(force_reload=True)
