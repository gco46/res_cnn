# coding=utf-8
import os
import numpy as np
from PIL import Image


THRESHOLD = 0.8
LABEL_D = {'ips': (0, 1, 2, 3), 'melanoma': (0, 1)}
# ips: good -> 0, bad -> 1, bgd -> 2, others -> 3
# melanoma: background -> 0, tumor -> 1
FCN_IN = 224


def getFilelist(path, ext):
    """
    get files path which have specified extension as a list recursiveliy.
    path: str, directory path you want to search
    ext: str, extension

    output: list, the components are file path
    """
    t = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                t.append(os.path.join(root, file))
    return t


def load_dataset(img_list, mask_list, in_size, size, step, method, resolution):
    """
    load dataset from image path and mask path list.

    img_list: list, image path list, path is str
    mask_list: list,
    in_size: int,
    size: int,
    step: int,
    method: str,
    resolution: list of int,

    output: (array, array), X_train, y_train
            !!! channels last !!!
            the shape is (num_samples, img_dim), (num_samples, target_dim)
    """
    # データセットを特定
    if 'ips' in img_list[0]:
        dataset = 'ips'
    else:
        dataset = 'melanoma'

    # img と mask を一致させるためにファイル名でソート
    img_list.sort()
    mask_list.sort()

    X = []
    y = []
    for img_path, mask_path in zip(img_list, mask_list):
        # img, mask を一つずつ読み込んでcrop
        img_vecs, targets = crop_img(
            img_path, mask_path, method,
            in_size, size, step, resolution
        )
        X += img_vecs
        y += targets
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def image2label(mask):
    """
    convert mask image(.png) to label

    mask: array, mask image
         in ips data, mask shape is (row, column, channels)
         in melanoma, (row, column)

    output: matrix array, shape is (row, column), 2 dims
    """
    # 各チャネルを0 or 1 のバイナリにする
    mask_bin = mask // 255
    if len(mask.shape) == 3:
        # ips dataset
        # good -> 0, bad -> 1, bgd -> 2, others -> 3　とする
        img_label = mask_bin[:, :, 0] * 1 + \
            mask_bin[:, :, 1] * 2 + mask_bin[:, :, 2] * 3
        # よくわからないラベルを全てothers へ
        img_label[img_label == 0] = 4
        img_label[img_label > 3] = 4
        img_label = img_label - 1
    else:
        # melanoma dataset
        img_label = mask_bin
    return img_label


def calcTarget(m_patch, method, resolution, data):
    """
    return the target vector, for classification, regression or fcn.

    m_patch: matrix array, patch of mask converted to label,
             the shape is (size, size)
    method: 'classification', 'regression' or 'fcn'
    data: str, 'ips' or 'melanoma'

    output: vector array or np.int64 or False(bool),
            target vector or one class label (in classification)
            return False if patch is filled by 'others' label
    """

    if method == "regression":
        # calc regression label using data variable
        target = calcRegTarget(m_patch, resolution, data)
    elif method == "classification":
        h, w = m_patch.shape
        label = m_patch[h // 2, w // 2]
        if label == 3:
            target = False
        else:
            target = np.int64(label)
    else:
        # fcnのtargetをFCN_INに揃える
        target = patch_resize(m_patch, FCN_IN)
        target = m_patch.flatten()
    return target


def class_label_hist(m_patch, data):
    """
    count the number of class labels in patch, and return histograms

    m_patch: matrix array,
    data: str, now supported 'ips' and 'melanoma'

    output: vector array, the histogram of class labels, !! dtype = float32 !!
            the last elements of hist is ignore label.
    """
    # label 辞書読み込み
    labels = LABEL_D[data]
    hist = []
    for c in labels:
        hist.append((m_patch == c).sum())
    hist = np.asarray(hist)
    return hist.astype(np.float32)


def calcRegTarget(m_patch, resolution, data):
    """
    calcurate target histograms, multi resolution

    m_patch: matrix array,
    resolution: list of int,
    data: int, 'ips', 'melanoma'

    output: vector array or None(ips), target histograms
            the length is \sum (res_int**2 * num_classes)
            in ips dataset, discard patches which is almost filled by 'others'
    """
    result = []
    hist = class_label_hist(m_patch, data)
    if data == 'ips':
        # ips dataset
        # others が パッチの大部分を占めていた場合、そのパッチはTraining には使わない
        n = int(m_patch.size * THRESHOLD)
        if hist[-1] > n:
            return False

    for res_int in resolution:
        # resolution を一つずつみてtarget histogramをつくる
        target = calcRegTarget_oneRes(m_patch, res_int, data)
        result += target
    result = np.asarray(result)
    return result


def calcRegTarget_oneRes(m_patch, res_int, data):
    """
    calcurate one target histograms in one regression.

    m_patch: matrix array,
    res_int: int,
    data: str, 'ips' or 'melanoma'

    output: !! list !!, target histograms
            length is (resolution**2 * num_class)
    """
    # ips or melanoma 判定
    if data == 'ips':
        num_classes = 3
    else:
        num_classes = 2

    result = []
    # patchのサイズからresolutionで割る範囲を決定する
    h, w = m_patch.shape
    local_size_h, rest_h = divmod(h, res_int)
    local_size_w, rest_w = divmod(w, res_int)

    for h_num in range(res_int):
        for w_num in range(res_int):
            # patchをresolutionによってさらに小さいパッチに分ける
            patch = m_patch[local_size_h * h_num:local_size_h * (h_num + 1),
                            local_size_w * w_num:local_size_w * (w_num + 1)]
            hist = class_label_hist(patch, data)
            if data == 'ips':
                # ips dataset
                # others のラベルを省いてヒストグラムを作る
                n = int(patch.size * THRESHOLD)
                if hist[-1] > n:
                    # othersが多ければ0とする
                    result.append([0., 0., 0.])
                    continue
                # histogram 正規化
                hist = hist[:-1] / np.sum(hist[:-1])
                result.append(hist)
            else:
                # melanoma dataset
                # histogram 正規化
                hist = hist / np.sum(hist)
                result.append(hist)

    result = np.asarray(result)
    result = result.flatten()
    assert result.size == res_int**2 * num_classes
    return list(result)


def crop_img(img_path, mask_path, method, in_size, size, step, resolution):
    """
    crop patches from original images and masks
    img_path: str, path to image
    mask_path: str or None, path to mask
    method: str,
    in_size: int,
    size: int,
    step: int,
    resolution: list,

    output: (list, list), first one is list of patch vectors
                          second is target vectors
            if mask_path is None, output is (list, None)
    """
    # imgとmaskが一致しているか確認
    img = img_path.split("/")[-1]
    img, _ = os.path.splitext(img)
    mask = mask_path.split("/")[-1]
    mask, _ = os.path.splitext(mask)
    assert img == mask, "file names are different"

    # ips or melanoma 判定
    if 'ips' in img_path:
        data = 'ips'
    else:
        data = 'melanoma'

    # img, mask読み込み
    # (row, column, channels)
    img = np.array(Image.open(img_path), dtype=np.float32)
    mask = np.array(Image.open(mask_path), dtype=int)
    mask = image2label(mask)
    h, w, c = img.shape

    img_vecs = []
    target_list = []
    for i in range((h - size) // step + 1):
        for j in range((w - size) // step + 1):
            patch = img[i * step:(i * step) + size,
                        j * step:(j * step) + size, :]
            m_patch = mask[i * step:(i * step) + size,
                           j * step:(j * step) + size]
            # deciding target from tmp
            target = calcTarget(m_patch, method, resolution, data)
            if not isinstance(target, bool):
                # targetに値が返っていれば、出力リストに加える
                if in_size != size:
                    # リサイズ
                    patch = patch_resize(patch, in_size)
                img_vecs.append(patch.flatten())
                target_list.append(target)

    return img_vecs, target_list


def patch_resize(im, s):
    """
    resize patch image.
    im: rgb image array, (row, column, channels)
    s: int, resize to s x s image. in_size

    output: rgb image array, (s, s, channels)
    """
    im = Image.fromarray(np.uint8(im))
    im = im.resize((s, s))
    im = np.array(im, dtype=np.float32)
    return im
