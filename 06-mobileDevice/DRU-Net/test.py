import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_psnr
from tensorflow import keras

import data_prepare

model_path = "./models/model.h5"

if __name__ == "__main__":
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        test_x_files = data_prepare.get_test_files()

        for i in range(len(test_x_files)):
            input_img = cv2.imread(filename=test_x_files[i], flags=cv2.IMREAD_GRAYSCALE)
            test_x = input_img[np.newaxis, ...] / 255.0
            test_y = model.predict(test_x)
            output_img = test_y.reshape(input_img.shape)
            cv2.imwrite('output_img.png',output_img*255.0)
            cv2.imwrite('input_img.png', input_img)
            plt.subplot(121), plt.imshow(input_img,cmap ='gray'), plt.title('input')
            plt.subplot(122), plt.imshow(output_img,cmap ='gray'), plt.title('output')
            psnr = compare_psnr(input_img, output_img)
            print('psnr=',psnr)
            plt.show()
