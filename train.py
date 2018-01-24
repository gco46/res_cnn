# coding=utf-8
from tools import Patch_DataLoader
import tools as tl
# import ptools as ptl
import models
from generator import patch_generator
from models import softmax_sparse_crossentropy, sparse_accuracy
from models import distribution_cross_entropy
from models import hamming_distance
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
import keras.backend as K

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import timeit


def train_model(method, resolution, dataset, in_size, size, step, arch,
                opt, lr, epochs, batch_size, l2_reg, decay, border_weight):
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
    m_list = ['regression', 'classification', 'fcn',
              'fcn_norm', 'ce_dist', 'fcn_dist', 'hamming']
    if method not in m_list:
        raise ValueError()

    # データセットによるクラス数指定
    if 'ips' in dataset:
        num_classes = 3
    elif 'melanoma' in dataset:
        num_classes = 2
    else:
        raise ValueError("dataset must be ips or melanoma")

    # ネットワークの出力ユニット数指定
    if method not in ["regression", "ce_dist", "fcn_dist", "hamming"]:
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
        if method == 'regression':
            metrics = "mse"
            loss_f = "mean_squared_error"
        elif method == 'fcn_dist':
            metrics = sparse_accuracy
            loss_f = softmax_sparse_crossentropy
        elif method == "ce_dist":
            metrics = distribution_cross_entropy
            loss_f = distribution_cross_entropy
        elif method == "hamming":
            metrics = None
            loss_f = hamming_distance

    # weights ディレクトリ作成
    try:
        n = dataset[-1]
        os.makedirs("weights/valid_all/dataset_" + str(n))
    except FileExistsError:
        pass
    dir_path = os.path.join("weights/valid_all/dataset_" + str(n))

    # モデル読み込み
    if method == "fcn":
        arch = "FCN_8s"
        print("arch : ", arch)
        in_shape = (in_size, in_size, 3)
        model = models.FCN_8s(num_classes, in_shape, l2_reg, nopad=True)
    elif method == "fcn_norm":
        arch = "FCN_8s_norm"
        print("arch : ", arch)
        in_shape = (in_size, in_size, 3)
        model = models.FCN_8s_norm(num_classes, in_shape, l2_reg, nopad=True)
        # 今後の操作のためmethodを上書き
        method = "fcn"
    elif method == "fcn_dist":
        arch = "FCN_8s_dist"
        print("arch : ", arch)
        in_shape = (in_size, in_size, 3)
        model = models.FCN_8s_dist(
            num_classes, in_shape, l2_reg, out_num, nopad=True)
    else:
        print("arch :", arch)
        if arch == "vgg_p5":
            model = models.myVGG_p5(in_size, l2_reg, method, out_num)
        elif arch == "vgg_p4":
            model = models.myVGG_p4(in_size, l2_reg, method, out_num)
        elif arch == "vgg_p4_multi":
            model = models.myVGG_p4_multi(in_size, l2_reg, method, out_num)
        else:
            raise ValueError("unknown arch")

    # データのパス読み込み
    img_list, mask_list = tl.load_datapath(dataset, mode="train")
    test_img_list, test_mask_list = tl.load_datapath(dataset, mode="test")

    # インスタンス化はするが読み込みはあとで行う。
    DataLoader = Patch_DataLoader(
        img_list, mask_list, in_size, size, step, method, resolution,
        border_weight=border_weight
    )
    test_DL = Patch_DataLoader(
        test_img_list, test_mask_list, in_size, size, step, method,
        resolution
    )

    # optimizer指定、モデルコンパイル
    # loss関数が引数をとる場合と場合分け
    if method not in ["ce_dist", "hamming"]:
        if opt == "SGD":
            model.compile(loss=loss_f,
                          optimizer=SGD(lr=lr, momentum=0.9, decay=decay),
                          metrics=[]
                          )
        elif opt == "Adadelta":
            lr = 1.0
            decay = 0
            model.compile(loss=loss_f,
                          optimizer=Adadelta(),
                          metrics=[metrics]
                          )
        elif opt == "Adam":
            if method == "fcn_dist":
                model.compile(loss={"fcn_out": loss_f, "dist_out": "mse"},
                              loss_weights={"fcn_out": 0.2, "dist_out": 1.0},
                              optimizer=Adam(lr=lr, decay=decay),
                              metrics=[])
            else:
                model.compile(loss=loss_f,
                              optimizer=Adam(lr=lr, decay=decay),
                              metrics=[]
                              )
        else:
            raise ValueError("argument 'opt' is wrong.")
    else:
        if opt == "SGD":
            model.compile(loss=loss_f(resolution),
                          optimizer=SGD(lr=lr, momentum=0.9, decay=decay),
                          metrics=[]
                          )
        elif opt == "Adadelta":
            lr = 1.0
            decay = 0
            model.compile(loss=loss_f(resolution),
                          optimizer=Adadelta(),
                          metrics=[]
                          )
        elif opt == "Adam":
            model.compile(loss=loss_f(resolution),
                          optimizer=Adam(lr=lr, decay=decay),
                          metrics=[]
                          )
        else:
            raise ValueError("argument 'opt' is wrong.")

    print("train on " + dataset)
    start_time = timeit.default_timer()
    if method != "fcn" and method != "fcn_dist":
        # fcn以外は.fit()で学習
        if "ips" in dataset:
            if border_weight is not None:
                X_train, y_train, s_weight = DataLoader.load_data()
            else:
                X_train, y_train = DataLoader.load_data()
                s_weight = None
            print("data loaded.")
            X_train = X_train.reshape(X_train.shape[0], in_size, in_size, 3)
            X_train /= 255.
            if method == "classification":
                y_train = np_utils.to_categorical(
                    y_train, num_classes=num_classes)
            X_test, y_test = test_DL.load_data()
            X_test = X_test.reshape(X_test.shape[0], in_size, in_size, 3)
            X_test /= 255.
            if method == "classification":
                y_test = np_utils.to_categorical(
                    y_test, num_classes=num_classes)
            hist = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_data=(X_test, y_test),
                             sample_weight=s_weight,
                             verbose=1,
                             )
        else:
            steps_per_epoch = DataLoader.num_samples // batch_size
            hist = model.fit_generator(
                generator=patch_generator(
                    in_size, size, step, dataset, batch_size, "train",
                    resolution, method, 5
                ),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
            )
    else:
        # fcnはgeneratorで学習
        steps_per_epoch = DataLoader.num_samples // batch_size
        if "ips" in dataset:
            val_step = test_DL.num_samples // batch_size
            hist = model.fit_generator(
                generator=patch_generator(
                    in_size, size, step, dataset, batch_size, "train",
                    resolution, method),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=patch_generator(
                    in_size, size, step, dataset, batch_size, "test",
                    resolution, method),
                validation_steps=val_step,
                verbose=1
            )
        else:
            hist = model.fit_generator(
                generator=patch_generator(
                    in_size, size, step, dataset, batch_size, "train",
                    resolution, method, 10),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
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
    if "ips" in dataset:
        val_loss = hist.history["val_loss"]
    nb_epoch = len(loss)
    plt.figure()
    plt.plot(range(nb_epoch), loss, label="loss")
    if "ips" in dataset:
        plt.plot(range(nb_epoch), val_loss, label="val_loss")
    if method == "fcn_dist":
        val_fcn_out_loss = hist.history["val_fcn_out_loss"]
        val_dist_out_loss = hist.history["val_dist_out_loss"]
        plt.plot(range(nb_epoch), val_fcn_out_loss, label="val_fcn_loss")
        plt.plot(range(nb_epoch), val_dist_out_loss, label="val_dist_loss")
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(dir_path, "loss.png"))
    plt.close()


def train_fcn_model(dataset, opt, lr, epochs, batch_size, l2_reg, decay,
                    img_size, resize_input=False, resolution=None):
    """
    train fcn with whole image.
    dataset: str, "ips" or "melanoma" + 1 - 5
    opt: str,
    lr: float, learning rate
    batch_size: int,
    img_size: tuple, (in_height, in_width)
                when this argument is set, inputs of network are resized to
                fixed size
    resize_input: bool
    """
    in_h, in_w = img_size
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

    model = models.FCN_8s(num_classes, (in_h, in_w, 3), l2_reg)

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
    X_train, y_train = tl.make_fcn_input(
        in_w, in_h, num_classes, dataset, resize_input, mode="train"
    )
    X_test, y_test = tl.make_fcn_input(
        in_w, in_h, num_classes, dataset, resize_input, mode="test"
    )

    # training
    print("train on " + dataset)
    start_time = timeit.default_timer()
    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     #  validation_data=(X_test, y_test),
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
        file.write("img_size:" + str(img_size) + "\n")
        file.write("lr:" + str(lr) + "\n")
        file.write("epochs:" + str(epochs) + "\n")
        file.write("batch_size:" + str(batch_size) + "\n")
        file.write("TrainingTime:%.2f m\n" % elapsed_time)

    # train loss だけプロットして保存
    loss = hist.history["loss"]
    if "ips" in dataset:
        val_loss = hist.history["val_loss"]
    nb_epoch = len(loss)
    plt.figure()
    plt.plot(range(nb_epoch), loss, label="loss")
    if "ips" in dataset:
        plt.plot(range(nb_epoch), val_loss, label="val_loss")
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(dir_path, "loss.png"))
    plt.close()


if __name__ == '__main__':
    for i in range(3, 4):
        K.clear_session()
        dataset = "ips_" + str(i)
        train_model(
            method="classification",
            resolution=None,
            dataset=dataset,
            in_size=150,
            size=150,
            step=45,
            arch="vgg_p4",
            opt="Adam",
            lr=1e-4,
            epochs=15,
            batch_size=16,
            l2_reg=1e-4,
            decay=0,
            border_weight=None
        )
    # for i in range(5, 6):
    #     K.clear_session()
    #     dataset = "melanoma_" + str(i)
    #     train_fcn_model(
    #         dataset=dataset,
    #         opt="Adam",
    #         lr=1e-5,
    #         epochs=50,
    #         batch_size=1,
    #         l2_reg=5e-5,
    #         decay=0,
    #         img_size=(1000, 1000),
    #         resize_input=False
    #     )
