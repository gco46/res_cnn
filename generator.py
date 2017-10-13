# coding=utf-8
from tools import Patch_DataLoader
import tools as tl
import numpy as np
import gc
import os


def fcn_generator(in_size, size, step, dataset, batch_size, mode, subsets=3):
    """
    in_size: int,
    size: int,
    step: int,
    dataset: str, "ips" or "melanoma", "_1" to "_5"
    """
    img_list, mask_list = tl.load_datapath(dataset, mode=mode)

    nb_samples = len(img_list)
    while 1:
        # 全サンプルに対して番号をつけ、シャッフルして番号順に読み込むことで
        # 逐次読み込みを実現する
        # nb_samplesがsubsetsで割り切れなくてもおけ
        index = list(range(int(subsets))) * \
            np.ceil(nb_samples / subsets).astype(int)
        # 割り切れない場合に、余分なindexを除去する
        for i in range(len(index) - nb_samples):
            index.pop()

        np.random.shuffle(index)
        for i in range(int(subsets)):   # subset loop
            # indexの番号順にサンプルを読み込む
            # bool_maskでフラグが立った画像のみ取り出せる
            bool_mask = (np.array(index) == i)
            img_subset = np.array(img_list)[bool_mask]
            mask_subset = np.array(mask_list)[bool_mask]
            img_subset = img_subset.tolist()
            mask_subset = mask_subset.tolist()
            DataLoader = Patch_DataLoader(
                img_subset, mask_subset, in_size, size, step, "fcn", None
            )
            X_train, y_train = DataLoader.load_data()
            X_train, y_train = shuffle_samples(X_train, y_train)
            X_train = X_train.reshape(X_train.shape[0], in_size, in_size, 3)
            X_train /= 255.
            y_train = y_train.reshape(y_train.shape[0], in_size, in_size, 1)
            y_train = y_train.astype(np.int32)
            batch_loop = X_train.shape[0] // batch_size
            for j in range(batch_loop):     # batch loop
                x = X_train[j * batch_size: (j + 1) * batch_size, ...]
                y = y_train[j * batch_size: (j + 1) * batch_size, ...]
                yield x, y
            del X_train
            del y_train
            gc.collect()


def shuffle_samples(X, y):
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    X_result = np.zeros(X.shape)
    y_result = np.zeros(y.shape)
    for i in range(X.shape[0]):
        X_result[i, ...] = X[order[i], ...]
        y_result[i, ...] = y[order[i], ...]
    return X_result, y_result
