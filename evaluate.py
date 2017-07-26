# coding=utf-8
import tools as tl
import os
import numpy as np
from PIL import Image
import sys
from sklearn.metrics import confusion_matrix


def seg_main(model):
    """
    evaluate model with segmentation metrics.
    model: str, model name like 'regression/Adam/vgg_p4_size150'.
    """
