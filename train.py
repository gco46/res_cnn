# coding=utf-8
from tools import Patch_DataLoader
import tools as tl
# import ptools as ptl
import models
from generator import fcn_generator
from models import softmax_sparse_crossentropy, sparse_accuracy
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.models import model_from_json
import keras.backend as K

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
          if method is 'fcn', arch must be 'vgg_p5'
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
        if method == "classification":
            metrics = "accuracy"
            loss_f = "categorical_crossentropy"
        else:
            metrics = sparse_accuracy
            loss_f = softmax_sparse_crossentropy
        resolution = None
        out_num = num_classes
    else:
        out_num = 0
        for i in resolution:
            out_num += i**2 * num_classes
        metrics = "mse"
        loss_f = "mean_squared_error"

    # weights ディレクトリ作成
    try:
        n = dataset[-1]
        os.makedirs("weights/valid_all/dataset_" + str(n))
    except FileExistsError:
        pass
    dir_path = os.path.join("weights/valid_all/dataset_" + str(n))

    # データのパス読み込み
    img_list, mask_list = tl.load_datapath(dataset, mode="train")

    # インスタンス化はするが読み込みはあとで行う。
    DataLoader = Patch_DataLoader(
        img_list, mask_list, in_size, size, step, method, resolution
    )

    # モデル読み込み
    if arch == "vgg_p5":
        if method == "fcn":
            model = models.fcn_p5_full(num_classes)
        else:
            model = models.myVGG_p5(in_size, l2_reg, method, out_num)
    elif arch == "vgg_p4":
        if method == "fcn":
            raise ValueError("fcn has only vgg_p5 models")
        else:
            model = models.myVGG_p4(in_size, l2_reg, method, out_num)
    else:
        ValueError("now support only vgg_p5")

    # optimizer指定、モデルコンパイル
    if opt == "SGD":
        model.compile(loss=loss_f,
                      optimizer=SGD(lr=lr, momentum=0.9, decay=decay),
                      metrics=[metrics]
                      )
    elif opt == "Adadelta":
        lr = 1.0
        decay = 0
        model.compile(loss=loss_f,
                      optimizer=Adadelta(),
                      metrics=[metrics]
                      )
    elif opt == "Adam":
        model.compile(loss=loss_f,
                      optimizer=Adam(lr=lr, decay=decay),
                      metrics=[metrics]
                      )
    else:
        raise ValueError("argument 'opt' is wrong.")

    print("train on " + dataset)
    start_time = timeit.default_timer()
    if method != "fcn":
        # fcn以外は.fit()で学習
        X_train, y_train = DataLoader.load_data()
        print("data loaded.")
        X_train = X_train.reshape(X_train.shape[0], in_size, in_size, 3)
        X_train /= 255.
        if method == "classification":
            y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        hist = model.fit(X_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         )
    else:
        # fcnはgeneratorで学習
        steps_per_epoch = DataLoader.num_samples // batch_size
        hist = model.fit_generator(
            generator=fcn_generator(in_size, size, step, dataset, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )

    elapsed_time = (timeit.default_timer() - start_time) / 60.
    print("train on %s takes %.2f m" % (dataset, elapsed_time))

    # モデル保存
    # fcnはなぜかjsonが作れないため例外処理する
    try:
        json_string = model.to_json()
        with open(os.path.join(dir_path, "train_arch.json"), "w") as file:
            file.write(json_string)
    except ValueError:
        print("couldnt save json_file, skipped")
    finally:
        model.save_weights(os.path.join(
            dir_path, "train_weights.h5"), overwrite=True)

    # パラメータなどをresult.txtに保存
    with open(os.path.join(dir_path, "result.txt"), "w") as file:
        title = ["<<", method, arch, ">>"]
        title = " ".join(title)
        file.write(title + "\n")
        file.write("in_size, size, step:" + str((in_size, size, step)) + "\n")
        file.write("resolution:" + str(resolution) + "\n")
        file.write("lr:" + str(lr) + "\n")
        file.write("epochs:" + str(epochs) + "\n")
        file.write("batch_size:" + str(batch_size) + "\n")
        file.write("l2_reg:" + str(l2_reg) + "\n")
        file.write("decay:" + str(decay) + "\n")
        file.write("TrainingTime:%.2f m\n" % elapsed_time)

    # train loss だけプロットして保存
    loss = hist.history["loss"]
    nb_epoch = len(loss)
    plt.figure()
    plt.plot(range(nb_epoch), loss, label="loss")
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(dir_path, "loss.png"))
    plt.close()


def train_fcn_model(dataset, opt, lr, epochs, batch_size, l2_reg, decay):
    """
    train fcn with whole image.
    dataset: str, "ips" or "melanoma" + 1 - 5
    opt: str,
    lr: float, learning rate
    batch_size: int,
    """
    in_h, in_w = (1000, 1000)
    # データセットによるクラス数指定
    if 'ips' in dataset:
        num_classes = 3
    elif 'melanoma' in dataset:
        num_classes = 2
    else:
        raise ValueError("dataset must be ips or melanoma")

    # weights ディレクトリ作成
    try:
        n = dataset[-1]
        os.makedirs("weights/valid_all/dataset_" + str(n))
    except FileExistsError:
        pass
    dir_path = os.path.join("weights/valid_all/dataset_" + str(n))

    model = models.fcn_p5_image(num_classes, (in_h, in_w, 3))

    metrics = sparse_accuracy
    loss_f = softmax_sparse_crossentropy
    # optimizer指定、モデルコンパイル
    if opt == "SGD":
        model.compile(loss=loss_f,
                      optimizer=SGD(lr=lr, momentum=0.9, decay=decay),
                      metrics=[metrics]
                      )
    elif opt == "Adadelta":
        lr = 1.0
        decay = 0
        model.compile(loss=loss_f,
                      optimizer=Adadelta(),
                      metrics=[metrics]
                      )
    elif opt == "Adam":
        model.compile(loss=loss_f,
                      optimizer=Adam(lr=lr, decay=decay),
                      metrics=[metrics]
                      )
    else:
        raise ValueError("argument 'opt' is wrong.")

    # データ読み込み
    img_list, mask_list = tl.load_datapath(dataset, mode="train")
    DL = Patch_DataLoader(img_list, mask_list)
    X_train = np.zeros((len(img_list), in_h, in_w, 3)).astype(np.float32)
    y_train = np.zeros((len(img_list), in_h, in_w)) + num_classes
    y_train = y_train.astype(np.int32)
    n = 0
    for im, ma in zip(img_list, mask_list):
        img = np.array(Image.open(im), dtype=np.float32) / 255.
        mask = np.array(Image.open(ma), dtype=np.int32)
        mask = DL.image2label(mask)
        if in_h > img.shape[0]:
            offset = (in_h - img.shape[0]) // 2
            X_train[n, offset:offset + img.shape[0], :, :] = img[...]
            y_train[n, offset:offset + mask.shape[0], :] = mask[...]
        elif in_w > img.shape[1]:
            offset = (in_w - img.shape[1]) // 2
            X_train[n, :, offset:offset + img.shape[1], :] = img[...]
            y_train[n, :, offset:offset + mask.shape[1]] = mask[...]
        n += 1
    y_train = y_train.reshape(
        y_train.shape[0], y_train.shape[1], y_train.shape[2], 1
    )

    # training
    print("train on " + dataset)
    start_time = timeit.default_timer()
    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1)
    elapsed_time = (timeit.default_timer() - start_time) / 60.
    print("train on %s takes %.2f m" % (dataset, elapsed_time))

    # モデル保存
    # fcnはなぜかjsonが作れないため例外処理する
    try:
        json_string = model.to_json()
        with open(os.path.join(dir_path, "train_arch.json"), "w") as file:
            file.write(json_string)
    except ValueError:
        print("couldnt save json_file, skipped")
    finally:
        model.save_weights(os.path.join(
            dir_path, "train_weights.h5"), overwrite=True)

    # パラメータなどをresult.txtに保存
    with open(os.path.join(dir_path, "result.txt"), "w") as file:
        title = ["<<", "fcn-image", ">>"]
        title = " ".join(title)
        file.write(title + "\n")
        file.write("lr:" + str(lr) + "\n")
        file.write("epochs:" + str(epochs) + "\n")
        file.write("batch_size:" + str(batch_size) + "\n")
        file.write("TrainingTime:%.2f m\n" % elapsed_time)

    # train loss だけプロットして保存
    loss = hist.history["loss"]
    nb_epoch = len(loss)
    plt.figure()
    plt.plot(range(nb_epoch), loss, label="loss")
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(dir_path, "loss.png"))
    plt.close()


if __name__ == '__main__':
    for i in range(1, 6):
        K.clear_session()
        dataset = "melanoma_" + str(i)
        train_fcn_model(
            dataset=dataset,
            opt="Adam",
            lr=1e-4,
            epochs=15,
            batch_size=1,
            l2_reg=0,
            decay=0
        )
    # train_model(
    #     method="regression",
    #     resolution=[2],
    #     dataset="melanoma_1",
    #     in_size=150,
    #     size=150,
    #     step=45,
    #     arch="vgg_p4",
    #     opt="Adam",
    #     lr=1e-4,
    #     epochs=1,
    #     batch_size=16,
    #     l2_reg=0,
    #     decay=0
    # )
