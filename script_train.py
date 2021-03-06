# coding=utf-8
from train import train_model
from inference import test_model, save_TestTime_asFile
from evaluate import evaluate_model
import shutil
import os.path as osp
import keras.backend as K


def mv_dirs(model_path):
    for i in range(1, 6):
        shutil.move(
            "weights/valid_all/dataset_" + str(i),
            osp.join("weights", model_path, "dataset_" + str(i))
        )


def make_model_name(arch, size, res, fcn=False):
    size = "_size" + str(size)
    if fcn:
        model_name = "fcn" + size
        return model_name
    if res is None:
        suffix = "_center"
    elif len(res) == 1:
        if res[0] == 1:
            suffix = ""
        else:
            r = str(res[0])
            suffix = "_res" + r + "-" + r
    else:
        m = list(map(str, res))
        r = "".join(m)
        suffix = "_res" + r

    model_name = arch + size + suffix
    return model_name


# set parameters -------------------------------------------------------
data = "melanoma"

in_size = 150
size = [50, 100, 150, 300]
step = 35
resolution = [[1], [2], [5], [1, 2, 5], [1, 2, 3, 4, 5]]
method = "ce_dist"
majority = False
lr = 1e-4
opt = "Adam"
batch_size = 64
epochs = 15
decay = 0
l2_reg = 0
arch = "vgg_p4"
# -----------------------------------------------------------------------

for s in size:
    for r in resolution:
        if r is None and method not in ["fcn", "classification"]:
            raise ValueError(
                "resolution 'None' is available only for "
                "fcn or classification"
            )
        for i in range(1, 6):
            K.clear_session()
            if method == "fcn":
                model_name = make_model_name(arch, s, r, fcn=True)
            else:
                model_name = make_model_name(arch, s, r, fcn=False)
            mpath = osp.join(data, method, opt, model_name)
            dataset = data + "_" + str(i)
            print("\n")
            print("< model > ", model_name)
            print("< dataset >", dataset)
            print()
            train_model(
                method=method,
                resolution=r,
                dataset=dataset,
                in_size=in_size,
                size=s,
                step=step,
                arch=arch,
                opt=opt,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                l2_reg=l2_reg,
                decay=decay,
                border_weight=None,
                binary=majority
            )
            test_model(
                method=method,
                resolution=r,
                dataset=dataset,
                in_size=in_size,
                size=s,
                step=step,
                label_map=False
            )
        mv_dirs(mpath)
        save_TestTime_asFile(mpath)
        evaluate_model(mpath)
