import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_psnr
from tensorflow import keras

import data_prepare_fault

model_path = "./models/model.h5"

if __name__ == "__main__":
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        test_x_files, test_y_files = data_prepare_fault.get_train_files()
        files_len = len(test_x_files)
        test_len = int(files_len * 0.95)
        var_x,var_y = data_prepare_fault.get_val_data(test_x_files[test_len:], test_y_files[test_len:])
        print(var_x.shape)
        for i in range(var_x.shape[0]):
            input_img = var_x[i]
            print(input_img.shape)
            test_y = model.predict(input_img[np.newaxis,...,np.newaxis])
            output_img = test_y.reshape(input_img.shape)
            cv2.imwrite('output_img.png',output_img*255.0)
            cv2.imwrite('input_img.png', input_img*255.0)
            cv2.imwrite('true.png', var_y[i] * 255.0)
            plt.subplot(131), plt.imshow(input_img,cmap ='gray'), plt.title('input')
            plt.subplot(132), plt.imshow(output_img,cmap ='gray'), plt.title('output')
            plt.subplot(133), plt.imshow(var_y[i], cmap='gray'), plt.title('true')
            psnr1 = compare_psnr(output_img, var_y[i])
            psnr2 = compare_psnr(input_img, var_y[i])
            psnr3 = compare_psnr(input_img, output_img)
            print('psnr1=',psnr1,' psnr2=',psnr2,' psnr3=',psnr3)
            plt.show()
