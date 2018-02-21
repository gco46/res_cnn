# coding=utf-8
from keras.models import model_from_json
import models
from tools import Patch_DataLoader, ProbMapConstructer
import tools as tl
import timeit
import os
from PIL import Image
import numpy as np
from scipy.misc import imresize
import keras.backend as K


def test_model(method, resolution, dataset, in_size, size, step,
               label_map=False, model_path="valid", prob_out="fcn"):
    """
    inference
    method: str,
    dataset: str, 'ips' or 'melanoma' + '_1' to '_5'
    in_size: int,
    size: int,
    step: int,
    model_path: str, path to model path you want to test
    """
    if not method in ['regression', 'classification', 'fcn', 'fcn_norm',
                      'fcn_dist', 'ce_dist', 'hamming', 'fcn_pre']:
        raise ValueError()

    if method not in ["regression", "fcn_dist", "ce_dist", "hamming"]:
        resolution = None

    if 'ips' in dataset:
        num_classes = 3
    else:
        num_classes = 2
    if model_path != "valid":
        model_path = os.path.join(
            "weights", model_path, "dataset_" + dataset[-1])
    else:
        model_path = "weights/valid_all/dataset_" + dataset[-1]
    try:
        if method == "ce_dist":
            out_num = 0
            for i in resolution:
                out_num += i**2 * 3
            model = models.myVGG_p4(in_size, 0, method, out_num, test=True)
        else:
            model = model_from_json(
                open(os.path.join(model_path, "train_arch.json")).read())
    except FileNotFoundError:
        in_shape = (in_size, in_size, 3)
        if method == "fcn" or method == "fcn_pre":
            model = models.FCN_8s(num_classes, in_shape, 0, nopad=True,
                                  test=True)
        elif method == "fcn_norm":
            model = models.FCN_8s_norm(num_classes, in_shape, 0, nopad=True)
        else:
            out_num = 0
            for i in resolution:
                out_num += i**2 * num_classes
            model = models.FCN_8s_dist(
                num_classes, in_shape, 0, out_num, nopad=True)
    model.load_weights(os.path.join(model_path, "train_weights.h5"))

    # データ読み込み
    img_list, mask_list = tl.load_datapath(dataset, mode="test")
    DataLoader = Patch_DataLoader(
        img_list, mask_list, in_size, size, step, method, resolution,
        mode="test"
    )

    print("visualize the result of " + dataset)
    # 可視化画像を保存するためのディレクトリ作成
    if method == "ce_dist" and len(resolution) > 1:
        resolution = [resolution[-1]]
    make_vis_dirs(model_path, resolution)

    elapsed_time = 0.
    elapsed_map_time = 0.
    p_count = 0
    for img_path, mask_path in zip(img_list, mask_list):
        # 可視化画像の名前を取得
        file_name = img_path.split("/")[-1]
        file_name, ext = os.path.splitext(file_name)
        file_name = file_name + ".png"
        # データ読み込み
        patches, _ = DataLoader.crop_img(img_path, mask_path, to_array=True)
        height = DataLoader.height
        width = DataLoader.width
        patches = patches.reshape(patches.shape[0], in_size, in_size, 3)
        p_count += patches.shape[0]
        patches /= 255.
        # 推定
        start_time = timeit.default_timer()
        prob = model.predict(patches, batch_size=16)
        elapsed_time += timeit.default_timer() - start_time
        if isinstance(prob, list):
            if prob_out == "fcn":
                prob = prob[0]
            elif prob_out == "dist":
                prob = prob[1]
            else:
                raise ValueError("prob_out is wrong")
        if method == "ce_dist" and len(resolution) > 1:
            prob = prob[:, -resolution[-1]**2 * 3:]
        PMC = ProbMapConstructer(
            model_out=prob,
            size=size,
            step=step,
            origin_h=height,
            origin_w=width,
            label_map=label_map,
            data=dataset[:-2],
            resolution=resolution
        )
        elapsed_map_time += timeit.default_timer() - start_time
        PMC.save_InfMap(model_path, file_name)
    test_time = elapsed_time / len(img_list)
    test_time_p = elapsed_time / p_count
    test_map_time = elapsed_map_time / len(img_list)
    time_array = np.array([test_time, test_time_p, test_map_time])
    print("test on %s takes %.7f s" % (dataset, test_time))
    print("test on %s takes %.7f s" % (dataset, test_time_p))
    print("test on %s takes %.7f s" % (dataset, test_map_time))
    np.savetxt(os.path.join(model_path, "test_time.txt"), time_array)


def test_fcn_model(dataset, img_size, resize_input=False, model_path="valid"):
    """
    """
    in_h, in_w = img_size
    if 'ips' in dataset:
        num_classes = 3
    else:
        num_classes = 2
    if model_path != "valid":
        model_path = os.path.join(
            "weights", model_path, "dataset_" + dataset[-1])
    else:
        model_path = "weights/valid_all/dataset_" + dataset[-1]
    try:
        model = model_from_json(
            open(os.path.join(model_path, "train_arch.json")).read())
    except FileNotFoundError:
        model = models.FCN_8s(num_classes, (in_h, in_w, 3), 0, test=True)
    model.load_weights(os.path.join(model_path, "train_weights.h5"))

    # データ読み込み
    img_list, mask_list = tl.load_datapath(dataset, mode="test")

    print("visualize the result of " + dataset)
    # 可視化画像を保存するためのディレクトリ作成
    make_vis_dirs(model_path)

    elapsed_time = 0.
    for im in img_list:
        # 可視化画像の名前を取得
        file_name = im.split("/")[-1]
        file_name, ext = os.path.splitext(file_name)
        file_name = file_name + ".png"
        # データ読み込み
        in_img = np.zeros((1, in_h, in_w, 3)).astype(np.float32)
        im = Image.open(im)
        if resize_input:
            im = im.resize((in_w, in_h))
        img = np.array(im, dtype=np.float32) / 255.
        if in_h > img.shape[0]:
            offset = (in_h - img.shape[0]) // 2
            in_img[0, offset:offset + img.shape[0], :, :] = img[...]
        elif in_w > img.shape[1]:
            offset = (in_w - img.shape[1]) // 2
            in_img[0, :, offset:offset + img.shape[1], :] = img[...]
        else:
            in_img[0, ...] = img[...]
            offset = 0
        # 推定
        start_time = timeit.default_timer()
        pred = model.predict(in_img)
        elapsed_time += timeit.default_timer() - start_time
        pred = normalize_infmap(pred)

        if resize_input:
            pred = resample_infmap(pred)

        if in_h > img.shape[0]:
            result = pred[0, offset:offset + img.shape[0], :, :]
        elif in_w > img.shape[1]:
            result = pred[0, :, offset:offset + img.shape[1], :]
        else:
            result = pred[0, ...]
        PMC = ProbMapConstructer(result, data=dataset[:-2])
        PMC.save_InfMap(model_path, file_name)
    test_time = elapsed_time / len(img_list)
    print("test on %s takes %.7f m" % (dataset, test_time))


def resample_infmap(prob_map, img_h=1200, img_w=1600):
    result = np.zeros((1, img_h, img_w, 3))
    prob_map *= 255
    for c in range(prob_map.shape[-1]):
        tmp = Image.fromarray(np.uint8(prob_map[0, :, :, c]))
        tmp = tmp.resize((img_w, img_h))
        tmp = np.array(tmp, dtype=np.float32)
        result[:, :, :, c] = tmp[...] / 255.
    return result


def normalize_infmap(prob_map):
    n, h, w, c = prob_map.shape
    reshaped_prob = prob_map.reshape(-1, c)
    total = np.sum(reshaped_prob, axis=1)
    sum_axis = np.zeros(reshaped_prob.shape)
    for i in range(c):
        sum_axis[:, i] = total
    norm_prob = reshaped_prob / sum_axis
    result = norm_prob.reshape(n, h, w, c)
    return result


def make_vis_dirs(model_path, resolution=None):
    """
    make vis and label directories.
    model_path: str,
    resolution: list or None,
    """
    try:
        if isinstance(resolution, list) and len(resolution) > 1:
            for i in resolution:
                os.makedirs(
                    os.path.join(
                        model_path, "vis" + str(i)
                    )
                )
                os.makedirs(
                    os.path.join(
                        model_path, "label" + str(i)
                    )
                )
        else:
            os.makedirs(
                os.path.join(model_path, "vis")
            )
            os.makedirs(
                os.path.join(model_path, "label")
            )
    except FileExistsError:
        pass


if __name__ == '__main__':
    # params = [
    #     ("ips/ce_dist/Adam/vgg_p4_size250_res125", 250),
    # ]
    # for m_path, size in params:
    #     K.clear_session()
    #     test_time = []
    #     for i in range(1, 6):
    #         dataset = "ips_" + str(i)
    #         test_model(
    #             method="ce_dist",
    #             resolution=[1, 2, 5],
    #             dataset=dataset,
    #             in_size=150,
    #             size=size,
    #             step=45,
    #             label_map=False,
    #             model_path=m_path,
    #             prob_out=None
    #         )

        #     tmp = np.loadtxt(
        #         os.path.join("weights", m_path, "dataset_" + str(i), "test_time.txt"),
        #         )
        #     test_time.append(list(tmp))
        # test_time = np.array(test_time)
        # np.savetxt(os.path.join("weights", m_path, "test_time.txt"), test_time)

    for i in range(1, 6):
        K.clear_session()
        dataset = "ips_" + str(i)
        test_model(
            method="regression",
            resolution=[2],
            dataset=dataset,
            in_size=150,
            size=150,
            step=45,
            label_map=False,
            model_path="ips/regression/Adam/vgg_p4_size300_res2-2",
            prob_out=None
        )

    # for i in range(1, 6):
    #     dataset = "ips_" + str(i)
    #     test_fcn_model(
    #         dataset=dataset,
    #         img_size=(900, 1200),
    #         resize_input=True,
    #         model_path="valid"
    #     )
