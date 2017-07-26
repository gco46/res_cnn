# coding=utf-8
import tools as tl
import os
import numpy as np
from PIL import Image
import sys
from sklearn.metrics import confusion_matrix


def evaluate_model(model, data):
    """
    evaluate model with segmentation metrics.
    model: str, model name like 'regression/Adam/vgg_p4_size150'.
    data: str, ips or melanoma
    """
    if data == "ips":
        WEIGHT_PATH = "weights/ips/"
    elif data == "melanoma":
        WEIGHT_PATH = "weights/melanoma/"
    else:
        raise ValueError("'data' must be ips or melanoma curenntly.")
    # 評価指標のリストを用意して格納、表示
    jaccard = []
    dice = []
    tpr = []
    tnr = []
    accuracy = []
    for i in range(1, 6):
        dataset = "dataset_" + str(i)
        # one fold 評価
        j, d, tp, tn, acc = evaluate_one_fold(model, dataset)
        jaccard.append(j)
        dice.append(d)
        tpr.append(tp)
        tnr.append(tn)
        accuracy.append(acc)

    jaccard = np.asarray(jaccard)
    dice = np.asarray(dice)
    tpr = np.asarray(tpr)
    tnr = np.asarray(tnr)
    accuracy = np.asarray(accuracy)
    j_mean = np.mean(jaccard)
    j_std = np.std(jaccard)
    d_mean = np.mean(dice)
    d_std = np.std(dice)
    tp_mean = np.mean(tpr)
    tp_std = np.std(tpr)
    tn_mean = np.mean(tnr)
    tn_std = np.std(tnr)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    print("model : ", model)
    print("jaccard index : ", j_mean, "+- ", j_std)
    print("dice : ", d_mean, "+- ", d_std)
    print("tpr : ", tp_mean, "+-", tp_std)
    print("tnr : ", tn_mean, "+-", tn_std)
    print("accuracy : ", acc_mean, "+-", acc_std)
    path = WEIGHT_PATH + model
    result = np.array([[j_mean, j_std],
                       [d_mean, d_std],
                       [tp_mean, tp_std],
                       [tn_mean, tn_std],
                       [acc_mean, acc_std]
                       ])

    np.savetxt(os.path.join(path, "seg_result.txt"), result)


def evaluate_one_fold(directory, dataset, w_path):
    """
    evaluate the result of one fold, with segmentation metrics.
    directory: str, model directory path like 'regression/Adam/vgg_p4_size150'
    dataset: str, dataset_ 1 to 5
    w_path: str, path to weights directory

    output: tuple of float, segmentation scores
            (jaccard, dice, tpr, tnr, acc)
    """
    path = os.path.join(w_path, directory, dataset)
    ld = os.listdir(path)
    if "label" in ld:
        # 単一resolutionの場合はそのままlabelディレクトリ読み込み
        pred_path = tl.getFilelist(
            os.path.join(path, "label")
        )
    else:
        # multi resolutionの場合は、最も解像度が高いディレクトリを読み込み
        tmp = 0
        for d in ld:
            if "label" in d and int(d[-1]) > tmp:
                tmp = int(d[-1])
        di = "label" + str(tmp)
        pred_path = tl.getFilelist(
            os.path.join(path, di)
        )

    # true path読み込み
    true_path = []
    for line in open("dataset/test_mask" + dataset[-1] + ".txt", "r"):
        img_file = (line.strip())
        true_path.append(img_file)
    pred_path.sort()
    true_path.sort()

    jaccard = []
    dice = []
    tpr = []
    tnr = []
    acc = []
    for pred, true in zip(pred_path, true_path):
        pred_name, _ = os.path.splitext(pred)
        true_name, _ = os.path.splitext(true)
        assert pred_name == true_name

        y_pred = np.array(Image.open(pred), int)
        y_true = np.array(Image.open(true), int)
        y_true = tl.image2label(y_true)

        #  others の場所を除く。
        y_true[y_true >= 4] = 0
        y_true = modify(y_true)
        # out of region for evaluation
        oor = ~(y_true == 0) * 1
        y_pred = y_pred * oor
        j, d, tp, tn, a = evaluate_one_image(y_true, y_pred)
        jaccard.append(j)
        dice.append(d)
        tpr.append(tp)
        tnr.append(tn)
        acc.append(a)
    jaccard = sum(jaccard) / len(jaccard)
    dice = sum(dice) / len(dice)
    tpr = sum(tpr) / len(tpr)
    tnr = sum(tnr) / len(tnr)
    acc = sum(acc) / len(acc)
    return jaccard, dice, tpr, tnr, acc
