import pickle
from tqdm import tqdm
import pandas as pd
import sys
from PIL import Image
#from matplotlib.pyplot import imshow
#from matplotlib import pyplot as plt
#from matplotlib.colors import LogNorm

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
auto_conf_pkl = 'usefull_data.pkl'
np.random.seed(42)


# data preparation

def get_img_path(path, rev_dict_classes, img_name, img_classe):
    return os.path.join(path, rev_dict_classes[img_classe], img_name)


def get_dataframe(train_path):
    files = []
    rev_dict_classes = {}
    dict_classes = {}
    i = 0
    for d in os.listdir(train_path):
        d_path = os.path.join(train_path, d)
        if os.path.isdir(d_path):
            dict_classes[d] = i
            rev_dict_classes[i] = d
            files.extend([(img, i) for img in os.listdir(d_path)])
            i += 1
    df = pd.DataFrame(data=files, columns=['Image', 'Classe'])
    return df.sort_values('Image'), dict_classes, rev_dict_classes


def resize_image(img, ratio, new_dimension=(256, 256)):
    img = np.asarray(img)
    # black and white image
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)
    # get ride of last channel if 4 channels images
    if img.shape[2] == 4:
        img = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=2)
    resized = Image.fromarray(img).resize(new_dimension)
    # imshow(np.asarray(resized))
    return resized


def preprocess_images(dataframe, img_ratio, path, preprocessed_full_path, rev_dict_classes):
    images = dataframe['Image'].values
    classes = dataframe['Classe'].values

    #images = images[:100]

    print("preprocess_images", len(images), " PATH: ", path)
    preprocessed = []
    for i in tqdm(range(len(images))):
        img_path = get_img_path(path, rev_dict_classes, images[i], classes[i])

        preprocessed_img = resize_image(Image.open(img_path), img_ratio)

        # save preprocessed image localy
        preprocessed_img_path = os.path.join(preprocessed_full_path, images[i])
        preprocessed_img.save(preprocessed_img_path)

        preprocessed.append(np.asarray(preprocessed_img))
        if not(np.asarray(preprocessed_img).shape[2] == 3):
            print(np.asarray(preprocessed_img).shape,
                  images[i], rev_dict_classes[classes[i]])
    print(len(preprocessed))
    return np.asarray(preprocessed)


def finalize_train_dataset(x, y, num_classes, train_val_spit_ratio=.8):
    print("lenght of x : ", x.shape[0], " length of y : ", y.shape[0])
    assert x.shape[0] == y.shape[0]
    np.random.seed(42)
    p = np.random.permutation(x.shape[0])
    x = x[p]
    x = x / 255.
    y = to_categorical(y[p], num_classes=num_classes)
    threshold = int(x.shape[0] * train_val_spit_ratio)
    x_train = x[:threshold]
    y_train = y[:threshold]
    x_val = x[threshold:]
    y_val = y[threshold:]
    return x_train, y_train, x_val, y_val


def get_training_data(PATH, preprocess_path='preprocessed'):

    train_path = os.path.join(PATH, "train")
    df_train, dict_classes, rev_dict_classes = get_dataframe(train_path)
    preprocessed_full_path = os.path.join(PATH, preprocess_path)
    # preliminary test in notebook show that images are squared
    img_ratio = 1

    print("df loaded, img_ratio: ", img_ratio)
    if not(os.path.exists(preprocessed_full_path)):
        os.makedirs(preprocessed_full_path)
        preprocessed = preprocess_images(
            df_train, img_ratio, train_path, preprocessed_full_path, rev_dict_classes)
    else:
        # existing preprocessed images
        preprocessed = np.array([np.asarray(Image.open(os.path.join(preprocessed_full_path, img)))
                                 for img in df_train['Image'].values])

    x_train, y_train, x_val, y_val = finalize_train_dataset(
        preprocessed, df_train['Classe'].values, len(dict_classes))

    to_save = {
        'classes_dict': dict_classes,
        'classes_rev_dict': rev_dict_classes,
        'img_ratio': img_ratio
    }
    with open(auto_conf_pkl, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    valgen = ImageDataGenerator()
    valgen.fit(x_val)
    return datagen, valgen, x_train, y_train, x_val, y_val
    # return datagen, x_train, y_train, x_val, y_val


def get_test_data(PATH):
    print("get_test")
    PATH = PATH + 'test'
    images = os.listdir(PATH)
    with open(auto_conf_pkl, 'rb') as handle:
        conf = pickle.load(handle)
    x_test = preprocess_images(images, conf['img_ratio'], PATH)
    print(x_test.shape)

    return x_test


if __name__ == "__main__":
    print('start main')

    # Training logic
    get_test_data(PATH)
    # Test logic
    # test()
