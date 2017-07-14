# coding=utf-8
import os
import numpy as np
from PIL import Image


def load_datapath(dataset):
    """
    load data path from .txt file
    dataset: str, 'ips' or 'melanoma' + '_1' to '_5'
    output: (list, list), path list of train and test data
    """
    # dataset のパス指定して.txtからファイルパス読み込み
    img_txt = "train_data" + dataset[-1] + ".txt"
    mask_txt = "train_mask" + dataset[-1] + ".txt"
    img_txt = os.path.join("data", dataset[:-2], "dataset", img_txt)
    mask_txt = os.path.join("data", dataset[:-2], "dataset", mask_txt)
    img_list = []
    mask_list = []
    for line in open(img_txt, "r"):
        img_list.append(line.strip())
    for line in open(mask_txt, "r"):
        mask_list.append(line.strip())
    return img_list.sort(), mask_list.sort()


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


class Patch_DataLoader(object):
    label_d = {'ips': (0, 1, 2, 3), 'melanoma': (0, 1)}
    # ips: good -> 0, bad -> 1, bgd -> 2, others -> 3
    # melanoma: background -> 0, tumor -> 1

    def __init__(self,
                 img_list,
                 mask_list,
                 in_size,
                 size,
                 step,
                 method,
                 resolution,
                 threshold=0.8):
        """
        img_list: list, image path list, path is str
        mask_list: list,
        in_size: int,
        size: int,
        step: int,
        method: str,
        resolution: list of int,
        """
        # img と mask を一致させるためにファイル名でソート
        img_list.sort()
        mask_list.sort()
        self.img_list = img_list
        self.mask_list = mask_list
        self.in_size = in_size
        self.size = size
        self.step = step
        self.res = resolution
        self.method = method
        self.threshold = threshold
        # datasetを特定
        if 'ips' in img_list[0]:
            self.datatype = 'ips'
            self.num_classes = 3
        else:
            self.datatype = 'melanoma'
            self.num_classes = 2
        self.num_samples = self.count_samples()

    def count_samples(self):
        """
        count sample patches.(for fcn training)
        output: int, number of samples
        """
        num_samples = 0
        for img_path in self.img_list:
            img = Image.open(img_path)
            w, h = img.size
            y_axis = (h - self.size) // self.step + 1
            x_axis = (w - self.size) // self.step + 1
            num_samples += x_axis * y_axis
        return num_samples

    def load_data(self):
        """
        load dataset from image path and mask path list.

        output: (array, array), X_train, y_train
                !!! channels last !!!
                the shape is (num_samples, img_dim), (num_samples, target_dim)
        """
        X = []
        y = []
        for img_path, mask_path in zip(self.img_list, self.mask_list):
            # img, mask を一つずつ読み込んでcrop
            img_vecs, targets = self.crop_img(img_path, mask_path)
            X += img_vecs
            y += targets
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def crop_img(self, img_path, mask_path):
        """
        crop patches from original images and masks
        img_path: str, path to image
        mask_path: str or None, path to mask

        output: (list, list), first one is list of patch vectors
                              second is target vectors
                if mask_path is None, output is (list, None)
        """
        # 可読性のためsize, stepはローカル変数にしておく
        size = self.size
        step = self.step
        # imgとmaskが一致しているか確認
        img = img_path.split("/")[-1]
        img, _ = os.path.splitext(img)
        mask = mask_path.split("/")[-1]
        mask, _ = os.path.splitext(mask)
        assert img == mask, "file names are different"

        # img, mask読み込み
        # (row, column, channels)
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=int)
        mask = self.image2label(mask)
        h, w, c = img.shape
        # test用に元画像のサイズを取得しておく
        self.height = h
        self.width = w

        img_vecs = []
        target_list = []
        for i in range((h - size) // step + 1):
            for j in range((w - size) // step + 1):
                patch = img[i * step:(i * step) + size,
                            j * step:(j * step) + size, :]
                m_patch = mask[i * step:(i * step) + size,
                               j * step:(j * step) + size]
                # deciding target from tmp
                target = self.calcTarget(m_patch)
                if not isinstance(target, bool):
                    # targetに値が返っていれば、出力リストに加える
                    if self.in_size != size:
                        # リサイズ
                        patch = self.patch_resize(patch)
                    img_vecs.append(patch.flatten())
                    target_list.append(target)
        return img_vecs, target_list

    def image2label(self, mask):
        """
        convert mask image(.png) to label
        mask: array, mask image
             in ips data, mask shape is (row, column, channels)
             in melanoma, (row, column)

        output: matrix array, shape is (row, column), 2 dims
        """
        # 各チャネルを0 or 1 のバイナリにする
        mask_bin = mask // 255
        if self.datatype == 'ips':
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

    def calcTarget(self, m_patch):
        """
        return the target vector, for classification, regression or fcn.
        m_patch: matrix array, patch of mask converted to label,
                 the shape is (size, size)
        output: vector array or np.int64 or False(bool),
                target vector or one class label (in classification)
                return False if patch is filled by 'others' label
        """
        if self.method == "regression":
            target = self.calcRegTarget(m_patch)
        elif self.method == "classification":
            h, w = m_patch.shape
            label = m_patch[h // 2, w // 2]
            if label == 3:
                target = False
            else:
                target = np.int64(label)
        else:
            m_patch = self.patch_resize(m_patch)
            target = m_patch.flatten()
        return target

    def patch_resize(self, im):
        """
        resize patch image to in_size x in_size.
        im: rgb image array, (row, column, channels)

        output: rgb image array, (s, s, channels)
        """
        im = Image.fromarray(np.uint8(im))
        im = im.resize((self.in_size, self.in_size))
        im = np.array(im, dtype=np.float32)
        return im

    def class_label_hist(self, m_patch):
        """
        count the number of class labels in patch, and return histograms

        m_patch: matrix array,
        data: str, now supported 'ips' and 'melanoma'

        output: vector array, the histogram of class labels, !! dtype = float32 !!
                the last elements of hist is ignore label.
        """
        # label 辞書読み込み
        labels = self.label_d[self.datatype]
        hist = []
        for c in labels:
            hist.append((m_patch == c).sum())
        hist = np.asarray(hist)
        return hist.astype(np.float32)

    def calcRegTarget(self, m_patch):
        """
        calcurate target histograms, multi resolution

        m_patch: matrix array,
        output: vector array or None(ips), target histograms
                the length is \sum (res_int**2 * num_classes)
                in ips dataset, discard patches which is almost filled by 'others'
        """
        result = []
        hist = self.class_label_hist(m_patch)
        if self.datatype == 'ips':
            # ips dataset
            # others が パッチの大部分を占めていた場合、そのパッチはTraining には使わない
            n = int(m_patch.size * self.threshold)
            if hist[-1] > n:
                return False

        for res_int in self.res:
            # resolution を一つずつみてtarget histogramをつくる
            target = self.calcRegTarget_oneRes(m_patch, res_int)
            result += target
        result = np.asarray(result)
        return result

    def calcRegTarget_oneRes(self, m_patch, res_int):
        """
        calcurate one target histograms in one regression.

        m_patch: matrix array,
        res_int: int,

        output: !! list !!, target histograms
                length is (resolution**2 * num_classes)
        """
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
                hist = self.class_label_hist(patch)
                if self.datatype == "ips":
                    # ips dataset
                    # others のラベルを省いてヒストグラムを作る
                    n = int(patch.size * self.threshold)
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
        assert result.size == res_int**2 * self.num_classes
        return list(result)
