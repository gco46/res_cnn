# coding=utf-8
import tools as tl
import ptools as ptl
import models
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.models import model_from_json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import timeit
from PIL import Image
import numpy as np


def train_model(method, resolution, dataset, in_size, size, step, arch,
                opt, lr, epochs, batch_size, l2_reg, decay):
    """
    train models, and save weights and loss graph
    method: str: 'classification', 'regression' or 'fcn'
    resolution: list of int or None, resolution of patch
                if method is not 'resolution', this must be None
    dataset: 'ips' or 'melanoma' + '_1' to '_5'
    in_size: int, input size of network
    size: int, cropped patch size
    step: int, patch sampling step
    arch: str, network architecture
          if method is 'fcn', arch must be 'fcn_p5_full'
    opt: str, optimizer 'SGD' or 'Adam'
    lr: float, learning rate
    epochs: int, number of epochs to train
    batch_size: int, batch size
    l2_reg: float, l2 regularization value
    decay: float, weight decay

    output: None
    """
    if not method in ['regression', 'classification', 'fcn']:
        raise ValueError()

    # データセットによるクラス数指定
    if 'ips' in dataset:
        num_classes = 3
    elif 'melanoma' in dataset:
        num_classes = 2
    else:
        raise ValueError("dataset must be ips or melanoma")

    # ネットワークの出力ユニット数指定
    if method != 'regression':
        resolution = None
        out_num = num_classes
    else:
        out_num = 0
        for i in resolution:
            out_num += i**2 * num_classes

    # weights ディレクトリ作成
    try:
        os.makedirs(os.path.join("weights/valid_all", dataset))
    except FileExistsError:
        pass
    dir_path = os.path.join("weights/valid_all", dataset)

    # dataset のパス指定して.txtからファイルパス読み込み
    img_txt = "train_data" + dataset[-1] + ".txt"
    mask_txt = "train_mask" + dataset[-1] + ".txt"
    img_txt = os.path.join("data", dataset[:-2], "img", data_txt)
    mask_txt = os.path.join("data", dataset[:-2], "mask", data_txt)
    img_list = []
    mask_list = []
    for line in open(img_txt, "r"):
        img_list.append(line.strip())
    for line in open(mask_list, "r"):
        mask_list.append(line.strip())

    X_train, y_train = load_dataset()
