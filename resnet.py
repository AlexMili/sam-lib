from __future__ import division

import sys
import cv2

from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model
import numpy as np
import yaml
import scipy.io
import scipy.ndimage

from sam.models import load_config, sam_resnet, kl_divergence, correlation_coefficient, nss

config = None
m = None


def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img / np.max(img) * 255

    return img


def preprocess_singleimage(image, shape_r, shape_c):
    ims = np.zeros((1, shape_r, shape_c, 3))

    padded_image = padding(image, shape_r, shape_c, 3)
    ims[0] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims.transpose((0, 3, 1, 2))

    return ims


def generator_test_singleimage(image):
    global config

    b_s = 1
    nb_images = 1
    gaussian = np.zeros((b_s, config["nb_gaussian"], config["shape_r_gt"], config["shape_c_gt"]))

    counter = 0
    while True:
        yield [preprocess_singleimage(image, config["shape_r"], config["shape_c"]), gaussian]
        counter = (counter + b_s) % nb_images


def load(weights_path, config_path):
    global config, m

    config = yaml.load(open(config_path, 'r'))

    x = Input((3, config["shape_r"], config["shape_c"]))
    x_maps = Input((config["nb_gaussian"], config["shape_r_gt"], config["shape_c_gt"]))

    # Load config file in the model
    load_config(config_path)

    # Compile model
    m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
    print("Compiling SAM-ResNet")
    m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])

    print("Loading SAM-ResNet weights")
    m.load_weights(weights_path)


def predict(image):
    global m

    nb_imgs_test = 1
    predictions = m.predict_generator(generator_test_singleimage(image=image), nb_imgs_test)[0]
    res = postprocess_predictions(predictions[0][0], image.shape[0], image.shape[1])

    return res.astype(int)


if __name__ == '__main__':
    config_path = "./config.yml"
    weights_path = "weights/sam-resnet_salicon_weights.pkl"
    image_path = sys.argv[1]

    # Load the model
    load(weights_path, config_path)

    print("Predicting saliency maps for {}".format(image_path))
    original_image = cv2.imread(image_path)
    salMap = predict(original_image)
