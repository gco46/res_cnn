# coding=utf-8
import numpy as np
from scipy.misc import imresize


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
        size: int, cropped size
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

    def construct_InferenceMap(self, maps):
        """
        construct inference probabirity maps, for test images.
        maps: array or list of array,

        output: list of arrays
        """
        if isinstance(maps, list):
            pass
            # construct by each resolution -----------
        else:
            pass
            # construct inference map --------------

    def construct_oneInferenceMap(self, map_array):
        """
        construct inference probabirity map
        this method is part of 'construct_InferenceMap'
        this method is also used for single resolution
        map_array: array, array of patch map
                    the shape is (num_samples, size, size, num_classes)
        """
        output = np.zeros((self.h, self.w, map_array.shape[-1]))
        counter = np.zeros((self.h, self.w))
        count_filter = np.ones((self.size, self.size))
        # フィルタを動かす範囲の計算
        # wがx軸(横方向), hがy軸(縦方向)
        w_axis = (self.w - self.size) // self.step + 1
        h_axis = (self.w - self.size) // self.step + 1
        p_num = 0
        for i in range(h_axis):
            for j in range(w_axis):
                output[i * step:i * step + size, j * step:j *
                       step + size, :] += map_array[p_num, :, :, :]
                counter[i * step:i * step + size, j *
                        step:j * step + size] += count_filter[:, :]
                p_num += 1
        # counterで平均を取る
        # 0除算対策のため0の場所に1を代入する
        counter[counter == 0] = 1
        for n in range(prob.shape[-1]):
            output[:, :, n] /= counter[:, :]
        return output

    def construct_patchMaps(self, prob):
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
            prob = self.normalize_prob(prob)
            prob_map = self.resampling_map(prob)
            return prob_map
        elif flag == "regression":
            prob = prob.reshape(num_samples, num_local * self.num_classes)
            prob_map = self.restore_Map_allRes(prob)
            return prob_map
        else:
            prob_map = self.restore_patchMap(prob)
            return prob_map

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

    def resampling_map(self, prob_map):
        """
        resize the output of cnn, to crop size
        prob_map: array, the shape is
                         (nb_samples, output_size, output_size, num_classes)
                    !! prob maps must be normalized !!

        output: array, the shape is (nb_sample, self.size, self.size, 3)
        """
        result = []
        for n in range(prob_map.shape[0]):
            # サンプルごとにリサイズ
            # channelごとにリサイズして繋げる
            one_prob = np.zeros((self.size, self.size, self.num_classes))
            for c in range(self.num_classes):
                im = imresize(
                    prob_map[n, :, :, c], (self.size, self.size)
                )
                patch_sample[:, :, c] = im[:, :]
            # imresizeの戻り値は0-255のため、正規化
            result.append(one_prob / 255.)
        result = np.stack(result, axis=0)
        return result

    def restore_Map_allRes(self, prob):
        """
        calcurate Map for each Res.
        prob: matrix array, (num_samples, num_dims)
        output: list or array,
                list of patchMaps, each comp is (samples, size, size, 3)
                In single resolution case, return one array.
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
        if len(output) == 1:
            output = output[0]
        return output

    def restore_patchMap(self, prob):
        """
        restore patch map from probabirity.
        this method is part of 'restore_patchMap'
        this method is also used for 'classification' prob map.
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
