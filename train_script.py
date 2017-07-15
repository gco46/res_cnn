# coding=utf-8
from train import train_model
import shutil


def five_fold(method, resolution, data, in_size, size, step, arch,
              opt, lr, epochs, batch_size, l2_reg, decay):
    model_name = "/".join([method, opt, arch + "_" + str(size)])
    print(model_name)
    print()
    for i in range(1, 6):
        dataset = data + "_" + str(i)
        train_model(method, resolution, dataset, in_size, size, step, arch,
                    opt, lr, epochs, batch_size, l2_reg, decay)
        shutil.move(
            "weights/valid_all/dataset_" + str(i),
            os.path.join("weights", model_name, "dataset_" + str(i))
        )


if __name__ == '__main__':
    params = [150, 300]
    for size in params:
        five_fold("fcn",
                  None,
                  "melanoma",
                  224,
                  size,
                  35,
                  "vgg_p5",
                  "Adam",
                  1e-4,
                  15,
                  16,
                  0,
                  0)
