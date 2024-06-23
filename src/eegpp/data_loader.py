from .import header_lb
from .import params
from .import utils
import joblib
import os
from pathlib import Path
LABEL_MARKER = "EpochNo"
SEQ_MARKER = "Time"

LB_DICT = {'W': 0, 'W*': 1, 'NR': 2, 'NR*': 3, 'R': 4, 'R*': 5}

SEP_CHECKED = False
SEPERATOR = '\t'
def get_lbid(lb_text):
    try:
        lb_id = LB_DICT[lb_text]
    except:
        lb_id = len(LB_DICT)
    return lb_id
def load_labels(inp):
    global SEP_CHECKED, SEPERATOR
    fin = open(inp, errors='ignore')

    labels = []
    times = []
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith(LABEL_MARKER):
            break
    if line == "":
        return [], []
    ic = 0
    while True:
        line = fin.readline()
        if line == "":
            break
        if not SEP_CHECKED:
            if line.__contains__(","):
                SEPERATOR = ","
            SEP_CHECKED = True
        if line == "\n":
            continue
        parts = line.split(SEPERATOR)
        # print(parts)
        epoch_id = int(parts[0])
        lb_text = parts[1]
        time_text = parts[2]
        label_v = get_lbid(lb_text)
        time_v = utils.convert_time(time_text)
        labels.append([label_v, epoch_id])
        times.append(time_v)
    fin.close()
    return labels, times, LB_DICT

def load_seq_data_only(inp, step=4000):
    global SEP_CHECKED, SEPERATOR
    fin = open(inp, encoding='utf-8', errors='ignore')
    misc = {}
    misc["BASE_NAME"] = Path(inp).stem
    headers = []
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith(SEQ_MARKER):
            break

        headers.append(line)

    mx1, mx2, mx3 = -10000, -10000, -10000
    mxs = [mx1, mx2, mx3]

    value_seqs = [[], [], []]
    c_seqs = [[], [], []]
    ctime = -1
    time_anchors = []
    ic = 0
    while True:
        line = fin.readline()
        ic += 1
        if ic % 100 == 0:
            print("\r%s" % ic, end="")
        if line == "":
            break
        if ic == 1 or not SEP_CHECKED:
            if line.__contains__(","):
                SEPERATOR = ","
                print("Sep: commas,")
            SEP_CHECKED = True
        line = line.strip()
        parts = line.split(SEPERATOR)
        time_text = parts[0]

        value_texts = parts[1:4]
        time_v = utils.convert_time(time_text)
        if time_v >= ctime + step:
            if ctime == -1:
                for i, value_text in enumerate(value_texts):

                    v = float(value_text)
                    if abs(v) > mxs[i]:
                        mxs[i] = abs(v)
                    c_seqs[i].append(v)
                ctime = time_v
                time_anchors.append(time_text)
                continue
            else:
                for i in range(3):
                    value_seqs[i].append(c_seqs[i][:params.MAX_SEQ_SIZE])
                c_seqs = [[], [], []]
                ctime = time_v
                time_anchors.append(time_text)


        for i, value_text in enumerate(value_texts):

            v = float(value_text)
            if abs(v) > mxs[i]:
                mxs[i] = abs(v)
            c_seqs[i].append(v)
    print("\nLast time text: ", time_text, time_anchors[-1])
    misc["TIME_ANCHORS"] = time_anchors
    misc["HEADER"] = "".join(headers) + header_lb.HEADER_2
    misc["LB_DICT"] = LB_DICT
    misc["mxs"] = mxs
    print("\nFinal Length: ", len(value_seqs[0]), len(misc["TIME_ANCHORS"]))
    return value_seqs, mxs, misc



def load_seq_data_with_labels(times, labels, inp):
    global SEP_CHECKED, SEPERATOR
    fin = open(inp, encoding='utf-8', errors='ignore')
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
    mx1, mx2, mx3 = -10000, -10000, -10000
    mxs = [mx1, mx2, mx3]
    while not is_exit:
        while time_v < times[cid]:
            line = fin.readline()
            if not SEP_CHECKED:
                if line.__contains__(","):
                    SEPERATOR = ","
                SEP_CHECKED = True
            if line == "":
                is_exit = True
                break
            parts = line.split(SEPERATOR)
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
            if len(value_texts) != 3:
                print(cid, line, value_texts)
                is_exit = True
                break

            for i, value_text in enumerate(value_texts):
                v = float(value_text)
                if abs(v) > mxs[i]:
                    mxs[i] = abs(v)

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
    label_seqs = label_seqs[:len(value_seqs[0])]
    assert len(value_seqs[0]) == len(label_seqs), "%s_%s" % (len(value_seqs[0]), len(label_seqs))
    misc = {}
    misc["LB_DICT"] = LB_DICT
    return value_seqs, label_seqs, mxs, misc


def load_data_with_labels(force_reload=False, dump_path=None, label_path=None, seq_path=None):
    if os.path.exists(dump_path) and force_reload is False:
        value_seqs, label_seqs, mxs, misc = joblib.load(dump_path)
        print(misc["LB_DICT"])
    else:
        labels, times, lb_dict = load_labels(label_path)
        print("LB_DICT", lb_dict)

        value_seqs, label_seqs, mxs, misc = load_seq_data_with_labels(times, labels, seq_path)
        joblib.dump((value_seqs, label_seqs, mxs, misc), dump_path)
    print(len(label_seqs), len(value_seqs), mxs)
    for i in range(len(label_seqs)):
        # print(len(value_seqs[0]), type(value_seqs[0]))

        assert len(value_seqs[0][i]) == params.MAX_SEQ_SIZE
        # print(len(value_seqs[1]), type(value_seqs[1]))
        assert len(value_seqs[1][i]) == params.MAX_SEQ_SIZE
        assert len(value_seqs[2][i]) == params.MAX_SEQ_SIZE


def load_data_no_label(force_reload=False, dump_path=None, seq_path=None, time_step=4000):
    if os.path.exists(dump_path) and force_reload is False:
        value_seqs, label_seqs, mxs, misc = joblib.load(dump_path)
        print(misc["LB_DICT"])
    else:
        value_seqs, mxs, misc = load_seq_data_only(seq_path, time_step)
        joblib.dump((value_seqs, mxs, misc), dump_path)
def force_load_all_with_labels():
    import yaml
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    data_dir = config["datasets"]["data_dir"]
    for i, seq_file in enumerate(config["datasets"]["seq_files"]):
        seq_file = "%s/%s" % (data_dir, seq_file)
        label_file = "%s/%s" % (data_dir, config["datasets"]["label_files"][i])
        dump_file = "%s/%s" % (data_dir, config["datasets"]["dump_files"][i])
        load_data_with_labels(force_reload=True, dump_path=dump_file, label_path=label_file, seq_path=seq_file)

def force_load_all_no_labels():
    import yaml
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    data_dir = config["datasets"]["data_dir"]
    tmp_dir = config["datasets"]["tmp_dir"]
    TIME_STEP = config["datasets"]["time_step"]

    for i, seq_file in enumerate(config["datasets"]["seq_files"]):
        seq_file = "%s/%s" % (data_dir, seq_file)
        dump_file = ("%s/%s" % (tmp_dir, seq_file)).replace(".txt", ".pkl")
        load_data_no_label(force_reload=True, dump_path=dump_file, seq_path=seq_file,time_step=TIME_STEP)


if __name__ == "__main__":
    force_load_all_with_labels()
    # force_load_all_no_labels()
    pass
