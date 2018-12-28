"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    with open(data_txt_file, "r") as label_file:
        image_name_list = []
        label_list = []
        for line in label_file.readlines():
            image_name_list.append(line.split(",")[0])
            label_list.append(int(line.split(",")[1].replace("\n", "")))

    data = {}
    image_list = []
    for image_name in image_name_list:
        image_list.append(io.imread(image_data_path + image_name + ".jpg"))
    data["image"] = np.array(image_list)
    label_len = len(label_list)
    label_temp = np.array(label_list).reshape((label_len, 1))
    data["label"] = label_temp

    return data
