# coding=utf-8
import numpy as np


class ProbMapConstructer(object):

    def __init__(self, model_out, size, step, origin_h, origin_w,
                 data, resolution):
        """
        model_out: array, output of model
                   the shape is (num_samples, num_outdim)
                   num_outdim equal to...
                   classification -> num_classes
                   regression -> \sum res_int**2 * num_classes
                   if fcn, (nub_samples, in_size, in_size, num_classes)
        size: int,
        step: int,
        origin_h: int, height of original image
        origin_w: int, width of original image
        data: str, 'ips' or 'melanoma'
        """
        # self.prob = model_out
        self.size = size
        self.step = step
        self.h = origin_h
        self.w = origin_w
        self.res = resolution
        self.datatype = data
        if data == 'ips':
            self.num_classes = 3
        else:
            self.num_classes = 2

    def construct_maps(self, prob):
        """
        normalize probabirity, and make patch probabirity map.
        output: array, patch prob map, (num_samples, )
        """
        # (?, num_classes)に変換
        if len(prob.shape) == 4:
            flag = "fcn"
            self.res = [1]
            num_samples = prob.shape[0]
            out_size = prob.shape[1]
            prob = prob.reshape(-1, self.num_classes)
        elif prob.shape[1] > self.num_classes:
            flag = "regression"
            num_samples = prob.shape[0]
            num_local = prob.shape[1] // self.num_classes
            prob = prob.reshape(num_samples * num_local, self.num_classes)
        else:
            self.res = [1]
            flag = "other"

        prob = self.normalize_prob(prob)

        if flag == "fcn":
            prob = prob.reshape(
                num_samples, out_size, out_size, self.num_classes
            )
            # resize to cropped size ----------
            return None
        elif flag == "regression":
            prob = prob.reshape(num_samples, num_local * self.num_classes)
            # restore each res------------------
        else:
            pass
            # restore ------------------

    def normalize_prob(self, reshaped_prob):
        """
        normalize prob,
        reshaped_prob: matrix array, the shape is (num_hist, num_classes)

        output: matrix array, normalized probs
        """
        # クラス推定値の値域を[0, 1)にするため，最小値が負のヒストグラムのみ最小値を引く
        min_axis = np.min(reshaped_prob, axis=1)
        min_bool = (min_axis < 0) * 1
        min_axis3 = np.zeros(reshaped_prob.shape)
        nb_classes = reshaped_prob.shape[1]
        for i in range(reshaped_prob.shape[1]):
            min_axis3[:, i] = min_axis * min_bool
        reshaped_prob = reshaped_prob - min_axis3

        # 各軸の和で正規化
        tmp = np.sum(reshaped_prob, axis=1)
        sum_axis = np.zeros(reshaped_prob.shape)
        for i in range(nb_classes):
            sum_axis[:, i] = tmp
        norm_prob = reshaped_prob / sum_axis
        return norm_prob

    def restore_Map_eatchRes(self, prob):
        """
        calcurate Map for each Res.
        prob: matrix array, (num_samples, num_dims)
        output: list, list of patchMaps
                if len(res) == 1,
        """
        output = []
        # skip でtmp_probを得るためのインデックスを調整
        skip = 0
        for count, res_int in enumerate(self.res):
            if count != 0:
                tmp_res = self.res[count - 1]
                skip += tmp_res**2 * self.num_classes
            # resolution一つずつtmp_prob に分割してpatchMapを得る
            tmp_prob = prob[:, skip:(res_int**2 * self.num_classes) + skip]
            output.append(self.restore_patchMap(tmp_prob))
        return

    def restore_patchMap(self, prob):
        """
        restore patch map from probabirity.
        prob: matrix array, (num_samples, num_dims)

        output: array, (num_samples, size, size, num_classes)
        """
        result = np.zeros(
            (prob.shape[0], self.size, self.size, self.num_classes)
        )
        tmp_map = np.zeros((self.size, self.size, self.num_classes))
        num_local = prob.shape[1] // self.num_classes
        res = int(np.sqrt(num_local))
        l_size = self.size // res
        for index in range(prob.shape[0]):  # sample loop
            for y in range(res):            # local resolution loop axis_y
                for x in range(res):        # local resolution loop axis_x
                    t = res * y + x
                    tmp_map[x * l_size:(x + 1) * l_size,
                            y * l_size:(y + 1) * l_size, :] \
                        = prob[index, 3 * t:3 * (t + 1)]
            result[index, :, :, :] = tmp_map[:, :, :]
        return result
