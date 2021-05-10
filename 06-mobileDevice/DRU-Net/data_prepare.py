import os

import cv2
import numpy as np
import random

train_x_dir = './data/val_blur'
train_y_dir = './data/val_sharp'
test_x_dir = './data/test_blur'

input_x = 240
input_y = 320


def get_train_files():
    train_folder = os.listdir(train_x_dir)
    train_x_files = []
    train_y_files = []
    for folder in train_folder:
        folder_path = os.path.join(train_x_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                train_x_file = os.path.join(folder_path, file)
                train_y_file = train_x_file.replace(train_x_dir, train_y_dir)
                train_x_files.append(train_x_file)
                train_y_files.append(train_y_file)
    return train_x_files, train_y_files


def get_test_files():
    test_folder = os.listdir(train_x_dir)
    test_x_files = []
    for folder in test_folder:
        folder_path = os.path.join(train_x_dir, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                test_x_file = os.path.join(folder_path, file)
                test_x_files.append(test_x_file)
    return test_x_files


def get_val_data(val_x_files, val_y_files):
    batch_x = []
    batch_y = []
    for i in range(len(val_x_files)):
        batch_x.append(cv2.imread(filename=val_x_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0)
        batch_y.append(cv2.imread(filename=val_y_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0)
    return np.array(batch_x), np.array(batch_y)


def train_datagen(train_x_files, train_y_files,
                  file_size=1):
    while (True):
        idx = list(range(len(train_x_files)))
        random.shuffle(idx)
        for i in idx:
            batch_x = []
            batch_y = []
            for j in range(file_size):
                clipImg(cv2.imread(filename=train_x_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0, batch_x)
                clipImg(cv2.imread(filename=train_y_files[i], flags=cv2.IMREAD_GRAYSCALE) / 255.0, batch_y)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            yield batch_x, batch_y


def clipImg(img, batch):
    for i in range(0, img.shape[0] - input_x + 1, input_x):
        for j in range(0, img.shape[1] - input_y + 1, input_y):
            batch.append(img[i:i + input_x,j:j + input_y])


if __name__ == '__main__':
    get_train_files()
