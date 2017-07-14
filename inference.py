# coding=utf-8
from keras.models import model_from_json
import models
from tools import Patch_DataLoader
import tools as tl


def test_model(method, dataset, in_size, size, step, resolution,
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
    if 'ips' in dataset:
        num_classes = 3
    else:
        num_classes = 2
    if model_path != "valid":
        model_path = "weights/" + model_path
    else:
        model_path = "weights/valid_all"
    try:
        model = model_from_json(
            open(os.path.join(model_path, dataset,
                              "train_arch.json")).read())
    except FileNotFoundError:
        model = models.fcn_p5_full(num_classes)
    model.load_weights(os.path.join(
        model_path, dataset, "train_weights.h5"))

    # データ読み込み
    img_list, mask_list = tl.load_datapath(dataset)
    DataLoader = Patch_DataLoader(
        img_list, mask_list, in_size, size, step, method, resolution
    )

    print("visualize the result of " + dataset)
    try:
        if isinstance(resolution, list):
            for i in resolution:
                os.makedirs(
                    os.path.join(
                        model_path, dataset,
                        "hsv" + str(i[0])
                    )
                )
                os.makedirs(
                    os.path.join(
                        model_path, dataset,
                        "label" + str(i[0])
                    )
                )
        else:
            os.makedirs(
                os.path.join(model_path, dataset, "hsv")
            )
            os.makedirs(
                os.path.join(model_path, dataset, "label")
            )
    except FileExistsError:
        pass
    hsv_path = os.path.join(model_path, dataset, "hsv")
    label_path = os.path.join(model_path, dataset, "label")
    start_time = timeit.default_timer()

    for img_path, mask_path in zip(img_list, mask_list):
        # 可視化画像の名前を取得
        file_name = img_path.split("/")[-1].replace("JPG", "png")
        patches, _ = DataLoader.crop_img(img_path,)
