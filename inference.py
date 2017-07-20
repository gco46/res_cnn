# coding=utf-8
from keras.models import model_from_json
import models
from tools import Patch_DataLoader
import tools as tl
from ptools import ProbMapConstructer
import timeit
import os


def test_model(method, resolution, dataset, in_size, size, step,
               model_path="valid"):
    """
    inference
    method: str,
    dataset: str, 'ips' or 'melanoma' + '_1' to '_5'
    in_size: int,
    size: int,
    step: int,
    model_path: str, path to model path you want to test
    """
    if method != "regression":
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
        model = model_from_json(
            open(os.path.join(model_path, "train_arch.json")).read())
    except FileNotFoundError:
        model = models.fcn_p5_full(num_classes)
    model.load_weights(os.path.join(model_path, "train_weights.h5"))

    # データ読み込み
    img_list, mask_list = tl.load_datapath(dataset)
    DataLoader = Patch_DataLoader(
        img_list, mask_list, in_size, size, step, method, resolution
    )

    print("visualize the result of " + dataset)
    # 可視化画像を保存するためのディレクトリ作成
    try:
        if isinstance(resolution, list):
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

    start_time = timeit.default_timer()
    for img_path, mask_path in zip(img_list, mask_list):
        # 可視化画像の名前を取得
        file_name = img_path.split("/")[-1].replace("JPG", "png")
        # データ読み込み
        patches, _ = DataLoader.crop_img(img_path, mask_path, to_array=True)
        height = DataLoader.height
        width = DataLoader.width
        patches = patches.reshape(patches.shape[0], in_size, in_size, 3)
        patches /= 255.
        # 推定
        prob = model.predict(patches, batch_size=16)
        MapConst = ProbMapConstructer(
            model_out=prob,
            size=size,
            step=step,
            origin_h=height,
            origin_w=width,
            data=dataset[:-2],
            resolution=resolution
        )
        MapConst.save_InfMap(model_path, file_name)
        raise ValueError


if __name__ == '__main__':
    test_model(
        method="classification",
        resolution=[1, 2, 3],
        dataset="melanoma_1",
        in_size=100,
        size=150,
        step=35
    )
