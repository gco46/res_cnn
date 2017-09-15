# coding=utf-8
from train import train_model
from inference import test_model
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


def make_model_name(arch, size, res):
    size = "_size" + str(size)
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


data = "ips"

in_size = 150
size = [50, 100, 150, 300]
step = 45
resolution = [[1], [2], [5], [1, 2, 5]]
lr = 1e-4
opt = "Adam"
batch_size = 16
epochs = 1
decay = 0
l2_reg = 0
arch = "vgg_p4"

for s in size:
    for r in resolution:
        if r is None:
            method = "classification"
        else:
            method = "regression"
        for i in range(1, 6):
            K.clear_session()
            model_name = make_model_name(arch, s, r)
            mpath = osp.join(data, method, opt, model_name)
            dataset = data + "_" + str(i)

            print()
            print()
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
                l2_reg=0,
                decay=0
            )
            test_model(
                method=method,
                resolution=r,
                dataset=dataset,
                in_size=in_size,
                size=s,
                step=step
            )
        mv_dirs(mpath)
        evaluate_model(mpath)
