import math
import os.path
import pathlib
from optparse import OptionParser
from pathlib import Path

import yaml

from . import params
from . import utils
from .dataset import EGGDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from tqdm import tqdm
import numpy as np
import joblib
from .get_model import get_model, TILE_SEQ, SIDE_FLAG, device
from .post_processing import correct_star, correct_4wr
import contextlib

CLASS_WEIGHT = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5, 0]).float().to(device)
CLASS_WEIGHT2 = None  # torch.tensor([2, 1, 0.1,  2, 2, 0.5]).float()

loss_function = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT)
loss_functionx = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, ignore_index=-1)
# def init_rd_seed():
torch.manual_seed(params.RD_SEED)
generator1 = torch.Generator().manual_seed(params.RD_SEED)


def get_model_dirname():
    return "out/M_EGG{}_EMG{}_MOT{}_RD{}".format(not params.OFF_EGG, not params.OFF_EMG, not params.OFF_MOT,
                                                 params.RD_SEED)


def parse_x():
    parser = OptionParser()
    # parser.add_option("-a", "--regg", dest="regg", action="store_true", help="advanced model")
    # parser.add_option("-b", "--remg", dest="remg", action="store_true", help="advanced model")
    # parser.add_option("-c", "--rmot", dest="rmot", action="store_true", help="advanced model")

    parser.add_option("-p", "--path", dest="path", type='string', default="None", help="yaml config file")
    parser.add_option("-e", "--clean", dest="clean", action="store_true", help="store tmp file")
    parser.add_option("-t", "--threshold", dest="threshold", type='float', default=params.STAR_THRESHOLD)
    parser.add_option("-n", "--norule", dest="norule", action="store_true")
    parser.add_option("-s", "--save", dest="save", action="store_true")
    parser.add_option("-v", "--visual", dest="visual", action="store_true")
    parser.add_option("-i", "--id", dest="id", type = 'string', default = "")
    parser.add_option("-l", "--silence", dest="silence", action="store_true")

    (cmd_options, args) = parser.parse_args()
    params.STAR_THRESHOLD = cmd_options.threshold

    params.OFF_MOT = True
    if cmd_options.norule:
        params.RULE = False
    if cmd_options.path != "None":
        params.DATA_CONFIG_PATH = cmd_options.path
    # print(cmd_options, params.OFF_EGG, params.OFF_EMG, params.OFF_MOT, params.RD_SEED)
    return cmd_options

# def get_loss_c3(out, lws_array, device, w3=None):
#     if w3 is None:
#         if params.WINDOW_SIZE == 3:
#             w3 = [0.85, 1, 0.85]
#         elif params.WINDOW_SIZE == 5:
#             w3 = [0.4, 0.85, 1, 0.85, 0.4]
#     target = lws_array.to(device)
#     loss = 0
#     for i in range(3):
#         lossi = loss_functionx(out[:, :, i], target[:, :, i])
#         loss += w3[i] * lossi
#     return loss


def get_dataset():
    import yaml
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    DATA_DIR = config["datasets"]["data_dir"]
    TMP_DIR = config["datasets"]["tmp_dir"]
    OUT_DIR = config["datasets"]["out_dir"]
    utils.ensureDir(TMP_DIR)
    utils.ensureDir(OUT_DIR)
    for i in range(len(config["datasets"]["seq_files"])):
        # full_dump_path = os.path.join(TMP_DIR, config["datasets"]["seq_files"][i]).replace(".txt", ".pkl")
        full_dump_path = Path(os.path.join(TMP_DIR, config["datasets"]["seq_files"][i])).with_suffix(".pkl")
        full_seq_path = os.path.join(DATA_DIR, config["datasets"]["seq_files"][i])
        print("Loading {}".format(full_dump_path))
        if not os.path.exists(full_dump_path):
            from .data_loader import load_data_no_label
            print("{} does not exist. Generating from {}".format(full_dump_path, full_seq_path))
            load_data_no_label(True, full_dump_path, full_seq_path)
        dataset = EGGDataset(dump_path=full_dump_path, tile_seq=TILE_SEQ, side_flag=SIDE_FLAG, with_label=False)
        yield dataset, dataset.idx_2lb
        # datasets.append([dataset, dataset.idx_2lb])


def infer(opts=None,fft=True):
    import yaml
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    OUT_DIR = config["datasets"]["out_dir"]
    SEPERATOR = "\t"
    if config["datasets"]["out_seperator"] == " ":
        SEPERATOR = " "

    # model_dir = get_model_dirname()
    # utils.ensureDir(model_dir)

    n_class = params.NUM_CLASSES
    model = get_model(n_class)
    model.load_state_dict(torch.load(params.MODEL_PATH, map_location=device))
    model.to(device)

    datasets = get_dataset()
    torch.no_grad()
    for i, ds in enumerate(datasets):
        print("Interring...")
        i = i + 1
        infer_ds, idx_2lb = ds
        print("Last time: %s\n" % infer_ds.misc["TIME_ANCHORS"][-1])

        BASE_NAME = infer_ds.misc["BASE_NAME"]
        dataloader = DataLoader(infer_ds, batch_size=params.BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)

        sm = torch.nn.Softmax(dim=-1)

        predicted_test = []
        predicted_binary = []
        ffts = []
        model.eval()
        all_x = []
        for ii, data in tqdm(enumerate(dataloader)):
            x, lbnamelb, _, lbws_array, _ = data
            if opts.save:
                all_x.append(x.detach().cpu().numpy())
            if fft:
                s = x.detach().numpy()[:, 0, params.MAX_SEQ_SIZE * int(params.WINDOW_SIZE/2): (int(params.WINDOW_SIZE/2)+1)*params.MAX_SEQ_SIZE]
                si = s*infer_ds.misc["mxs"][0]
                r = utils.get_fft(si)
                ffts.append(r)
                # if (ii%20==0 and ii>100):
                #     print(infer_ds.misc["TIME_ANCHORS"][ii*10], si[:, :10])

            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)

            x = x.float().to(device)
            prediction, prediction2 = model(x)
            predicted_test.append(prediction.detach().cpu())
            predicted_binary.append(prediction2.detach().cpu())


        if params.OUT_3C:
            predicted_test = torch.concat(predicted_test, dim=0).detach().cpu()[:, :-1, params.POS_ID]
            predicted_binary = torch.concat(predicted_binary, dim=0).detach().cpu()[:, :, params.POS_ID]

        else:
            raise 'Not implemented yet'
            predicted_test = torch.concat(predicted_test, dim=0).detach().cpu()[:, :-1]

        predicted_test = sm(predicted_test)
        predicted_binary = sm(predicted_binary)

        np.savetxt("%s/%s_SCORES.txt" % (OUT_DIR, BASE_NAME), predicted_test, fmt="%.12f")
        if opts.save:
            all_x = np.concatenate(all_x, axis=0)
            np.save("%s/%s_SIGNALS.npy" % (OUT_DIR, BASE_NAME), all_x)
        # np.savetxt("%s/%s_BINARY_SCORES.txt" % (OUT_DIR, BASE_NAME), predicted_binary, fmt="%.12f")
        predicted_lb_binary = np.argmax(predicted_binary, axis=-1)
        # fout_star_lb = open("%s/%s_LB_BIARY.txt" % (OUT_DIR, BASE_NAME), "w")
        # for lb_binary in predicted_lb_binary:
        #     if lb_binary == 0:
        #         fout_star_lb.write("-\n")
        #     else:
        #         fout_star_lb.write("*\n")
        # fout_star_lb.close()
        # predicted_lbids = np.argmax(ss.numpy(), axis=-1)

        predicted_lbids = correct_star(predicted_test.numpy(), params.STAR_THRESHOLD)
        if params.RULE:
            correct_4wr(predicted_lbids)
        predicted_lbs = []
        for lb_id in predicted_lbids:
            predicted_lbs.append(idx_2lb[lb_id])
        fout = open("%s/%s_LBTEXT.txt" % (OUT_DIR, BASE_NAME), "w")
        for lbname in predicted_lbs:
            fout.write("%s\n" % lbname)
        fout.close()
        if fft:
            fout2 = open("%s/%s_FULL_LABEL.txt" % (OUT_DIR, BASE_NAME), "w")
            fout2.write("%s\n" % infer_ds.misc["HEADER"])
            ffts = np.concatenate(ffts).reshape(-1, 240)
            for i in range(len(predicted_lbs)):
                fout2.write("%s\n" % SEPERATOR.join(["%s" % (i+1),  predicted_lbs[i], infer_ds.misc["TIME_ANCHORS"][i], utils.convert_array2str(ffts[i], sep=SEPERATOR)]))
            fout2.close()

        if opts.clean:
            os.remove(infer_ds.dump_path)
def plot(opts):
    print("In plotting mode: ...")
    config = yaml.safe_load(open(params.DATA_CONFIG_PATH))
    OUT_DIR = config["datasets"]["out_dir"]
    BASE_NAME = pathlib.Path(config["datasets"]["seq_files"][0]).stem
    all_x =  np.load("%s/%s_SIGNALS.npy" % (OUT_DIR, BASE_NAME))
    f =  open("%s/%s_LBTEXT.txt" % (OUT_DIR, BASE_NAME))
    predicted_lbs = [l.strip() for l in f.readlines()]
    from .visualization2 import plt, plot2ccla
    FIG_DIR = "%s/%s/figures" % (OUT_DIR, BASE_NAME)
    utils.ensureDir(FIG_DIR)
    print("Plotting to %s ..." % FIG_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def parse_ids(eid):
        if (eid.__contains__(",")):
            eids = eid.split(",")
            eids = [int(eid) - 1 for eid in eids]
        else:
            eids = [int(eid) - 1]
        return eids
    def ploteid(eid):
        seq = all_x[eid]
        if (eid >= params.POS_ID):
            ws_lbs = predicted_lbs[eid - params.POS_ID:eid + params.POS_ID + 1]
        else:
            ws_lbs = ["_" for _ in range(params.POS_ID - eid)] + predicted_lbs[0:eid + params.POS_ID + 1]

        plot2ccla(fig, axes, seq, ("E%7d_%s" % (eid + 1, ws_lbs[params.POS_ID])).replace(" ", "0"), ws_lbs, FIG_DIR)
    if len(opts.id) > 0:
        eids = parse_ids(opts.id)
        for eid in eids:
            print("Plotting epoch: %d" % (eid+1))
            ploteid(eid)
        exit(0)
    print("Enter EpochId with commas separator (Start indexing from 1)")
    print("Enter -1 to exit.")

    while True:
        eid = input()
        eids = parse_ids(eid)
        if eids[0] == -2:
            print("Exit.")
            exit(0)
        for eid in eids:
            print("Plotting epoch: %d" % (eid+1))
            ploteid(eid)

def infer_cmd():
    opts = parse_x()
    import sys
    if opts.silence:
        f = open(os.devnull, 'w')
        sys.stdout = f

    print(opts)
    if opts.visual:
        plot(opts=opts)
    else:
        infer(opts=opts)
if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    infer_cmd()
