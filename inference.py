# coding=utf-8
from keras.models import model_from_json
import models
from tools import Patch_DataLoader


def test_model(method, dataset, in_size, size, step,
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

    img_txt = "train_data" + dataset[-1] + ".txt"
    img_txt = os.path.join("data", dataset[:-2], "dataset", img_txt)
    img_list = []
    for line in open(img_txt, "r"):
        img_list.append(line.strip())
