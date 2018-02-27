# coding=utf-8
import gc
import numpy as np
from keras.utils import np_utils
from tools import Patch_DataLoader
import tools as tl


def patch_generator(in_size, size, step, dataset, batch_size, mode,
                    resolution, method, subsets=3):
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
                img_subset, mask_subset, in_size, size, step, method,
                resolution
            )
            X_train, y_train = DataLoader.load_data()
            X_train, y_train = shuffle_samples(X_train, y_train)
            X_train = X_train.reshape(
                X_train.shape[0], in_size, in_size, 3)
            X_train /= 255.
            if method == "fcn":
                y_train = y_train.reshape(
                    y_train.shape[0], in_size, in_size, 1)
                y_train = y_train.astype(np.int32)
            elif method == "classification":
                if "melanoma" in dataset:
                    num_classes = 2
                else:
                    num_classes = 3
                y_train = np_utils.to_categorical(
                    y_train, num_classes=num_classes)

            batch_loop = X_train.shape[0] // batch_size

            for j in range(batch_loop):     # batch loop
                x = X_train[j * batch_size: (j + 1) * batch_size, ...]
                y = y_train[j * batch_size: (j + 1) * batch_size, ...]
                yield x, y
                # del x, y
                gc.collect()
            # del X_train
            # del y_train
            # gc.collect()


def shuffle_samples(*args):
    """
    shuffle more than one 1-d arrays, associating the order.
    arrays must be same length.
    output: tuple, the length equals to the number of input arguments.
    """
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled = list(zip(*zipped))
    result = []
    for ar in shuffled:
        result.append(np.asarray(ar))
    return result
