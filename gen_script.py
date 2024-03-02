import os


def gen_script():
    fout = open("runx.sh", "w")
    for regg in ["","--regg"]:
        for remg in ["", "--remg"]:
            for rmot in ["", "--rmot"]:
                if regg=="--regg" and remg=="--remg" and rmot == "--rmot":
                    continue
                fout.write("python train.py " + regg +" "+ remg +" "+ rmot+"\n")
                # for tid in ["1", "2", "3"]:
                #     fout.write("python xshap.py "+ regg +" "+ remg +" "+ rmot+
                #                " --train={}  --test={}\n".format(1, tid))
                #     fout.write("python visualization2.py "+ regg +" "+ remg +" "+ rmot+
                #                " --train={}  --test={}\n".format(1, tid))
                #     fout.write("python filter_figs.py "+ regg +" "+ remg +" "+ rmot+
                #                " --train={}  --test={}\n".format(1, tid))
                fout.write("#################\n")


    fout.close()
    os.system("chmod +x ./runx.sh")
if __name__ == "__main__":
    gen_script()