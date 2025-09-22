#from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 正則表達式套件
import re
import os
# 導入需要的程式庫
import tensorflow.keras as keras
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization as bn
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session  # 分配GPU會用到
import tensorflow.compat.v1 as tf
from tensorflow.keras.losses import categorical_crossentropy
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
# from Warning import *
from model import *
from keras.callbacks import EarlyStopping  # early stoppping
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import math
from keras.preprocessing.image import ImageDataGenerator
import cv2
import re
import gzip
import shutil
from keras.callbacks import Callback
from skimage.transform import resize
from keras.models import load_model
from keras.utils import normalize

import pydicom

class EpochSaver(Callback):
    def __init__(self, file_path):
        self.file_path = file_path
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        with open(self.file_path, 'a') as f:
            f.write(f"Epoch {epoch+1}:\n")
            f.write(f"{logs}\n")
            f.write("-"*50 + "\n")
                

img_path = []   # ct圖路徑
mask_path = []  # mask路徑

volume = "H:/TOPIC/Unet_liver_segmentation/tumor_data/1/resort/New_14筆/DICOMM/SE5/"
# segment = "H:/TOPIC/Unet_liver_segmentation/tumor_data/mask/New_14/DICOMM/"

imgpath = os.listdir(volume)  # img_aug
# maskpath = os.listdir(segment)  # mask_aug

imgpath = sorted(imgpath)
# maskpath = sorted(maskpath)

for img1 in imgpath:
    path = volume + img1  # img_aug
    img_path.append(path)
print(img_path)
# for img2 in maskpath:
#     path = segment + img2  # mask_aug
#     mask_path.append(path)
# print(mask_path)

# 讀取fold檔案，進行訓練和測試
for fold in range(1):
    
    liver_Dice_coef = 0
    liver_samples = 0
    eight_segmentation_Dice_coef = 0
    eight_segmentation_samples = 0

    # train_data = []
    # train_label = []
    test_data = []
    test_label = []
    eight_mask_train = []
    eight_mask_test = []
    
    new_size = (128, 128)
    image_dataset = []
    mask_dataset = []

    for path in img_path:
        # path = path.split(",") 
        # img_3D = nib.load(path).get_fdata()
        dicom_dataset = pydicom.dcmread(path)

        # Access DICOM metadata
        patient_name = dicom_dataset.PatientName
        study_description = dicom_dataset.StudyDescription
        # ... and so on

        # Access DICOM pixel data
        pixel_array = dicom_dataset.pixel_array
        img_3D = np.array(pixel_array)
        # img_3D = img_3D / 255.0
        img_3D = nib.Nifti1Image(img_3D, affine=np.eye(4))  # Assuming your image data is in a NumPy array
        print("np.shape(img_3D) : ", np.shape(img_3D))
        img_3D = img_3D.get_fdata()
        img_3D = np.fliplr(img_3D)
        img_min = 675  # minimum HU level  # 725
        img_max = 1400  # maximum HU level  # 1450
        img_3D[img_3D < img_min] = img_min  
        img_3D[img_3D > img_max] = img_max  
        img_3D = (img_3D - img_min) / ((img_max - img_min) * 1.0)  # * 255.0

        # mask_eight_3D[mask_eight_3D >= 1] = 1
        slice_img = resize(img_3D, new_size, order=0, anti_aliasing=False, preserve_range=True)
        print("np.shape(slice_img) : ", np.shape(slice_img))
        slice_img = np.expand_dims(slice_img, axis=0)  # 將 slice_img 轉換成形狀為 (1, 128, 128) 的 numpy 陣列
        print("---train---")
        print("Image data shape is: ", np.shape(slice_img))
        print("Max pixel value in image is: ", slice_img.max())
        print("Min pixel value in image is: ", slice_img.min())

        image_dataset.append(slice_img)


    train_data = np.concatenate(image_dataset, axis=0)

    new_size = (128, 128)
    image_dataset = []
    mask_dataset = []
    # for path in mask_path:

    #     mask_3D = np.load(path)
    #     mask_3D = np.fliplr(mask_3D)
    #     mask_3D[mask_3D >= 1] = 1

    #     slice_mask = resize(mask_3D, new_size, order=0, anti_aliasing=False, preserve_range=True)
    #     slice_mask = np.expand_dims(slice_mask, axis=0)  # 將 slice_mask 轉換成形狀為 (1, 128, 128) 的 numpy 陣列
    #     print("np.shape(slice_mask) : ", np.shape(slice_mask))
    #     mask_dataset.append(slice_mask)

    # train_label = np.concatenate(mask_dataset, axis=0)
    # train_data = np.expand_dims(train_data, axis=-1)
    # train_label = np.expand_dims(train_label, axis=-1)

    print("np.shape(train_data) : ", np.shape(train_data))  # (:, 128, 128, 1)
    # print("np.shape(train_label) : ", np.shape(train_label))  # (:, 128, 128, 1)
        
    model = load_model(f'./h5_file/chang_gung_model_1.h5', compile=False)

    prediction_liver = model.predict(train_data)

    prediction_liver[prediction_liver >= 0.1] = 1
    prediction_liver[prediction_liver < 0.1] = 0

    print("np.shape(prediction_liver) : ", np.shape(prediction_liver))
    print("Image data shape is: ", np.shape(prediction_liver))

    print("=================================")
    
    background_color = 0
    image_area = prediction_liver.shape[1] * prediction_liver.shape[2]
    threshold_area = 0.01 * image_area  # 1% 圖像面積
    prediction_liver = prediction_liver.astype(np.uint8)
    
    # 找區域輪廓
    for p_slice in range(prediction_liver.shape[0]):
        
        contours, _ = cv2.findContours(prediction_liver[p_slice, :, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # traversal 每個輪廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < threshold_area:
                # 比threshold_area小，變背景
                cv2.drawContours(prediction_liver[p_slice, :, :, 0], [contour], 0, background_color, -1)

    # for s in range(len(prediction_liver)):
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = ((20, 20)))

    #     ax1.imshow(train_data[s], cmap = 'gray')
    #     ax1.set_title(f"{s} : train_data", fontsize = "x-large")
    #     ax1.grid(False)
    #     ax2.imshow(train_label[s], cmap = 'gray')
    #     ax2.set_title(f"train_label : {fold}", fontsize = "x-large")
    #     ax2.grid(False)
    #     ax3.imshow(prediction_liver[s], cmap = 'gray')
    #     ax3.set_title(f"{s} : prediction_liver", fontsize = "x-large")
    #     ax3.grid(False)
    #     # plt.savefig(f'H:/TOPIC/Unet_liver_segmentation/fold/segmentation_photo/fold{fold+1}/img_{i}_seg_{j}.png')
    #     plt.show()
    #     plt.close()
    model = load_model(f'./h5_file/chang_gung_model_region.hdf5', compile=False)
    region_pred = model.predict(prediction_liver)
    region_pred[region_pred>=0.3] = 1
    region_pred[region_pred<0.3] = 0
    print("np.shape(region_pred) : ", np.shape(region_pred))

    # fig, (ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(1, 11, figsize = ((20, 20)))
    # ax00.imshow(np.rot90(prediction_liver[0, :, :, 0], 3), cmap = 'gray')
    # ax0.set_title(f"0 : {fold}", fontsize = "x-large")
    # ax0.grid(False) 
    # ax0.imshow(np.rot90(region_pred[0, :, :, 0], 3), cmap = 'gray')
    # ax0.set_title(f"0 : {fold}", fontsize = "x-large")
    # ax0.grid(False) 
    # ax1.imshow(np.rot90(region_pred[0, :, :, 1], 3), cmap = 'gray')
    # ax1.set_title(f"1 : {fold}", fontsize = "x-large")
    # ax1.grid(False)
    # ax2.imshow(np.rot90(region_pred[0, :, :, 2], 3), cmap = 'gray')
    # ax2.set_title(f"2 : {fold}", fontsize = "x-large")
    # ax2.grid(False)
    # ax3.imshow(np.rot90(region_pred[0, :, :, 3], 3), cmap = 'gray')
    # ax3.set_title(f"3 : {fold}", fontsize = "x-large")
    # ax3.grid(False)
    # ax4.imshow(np.rot90(region_pred[0, :, :, 4], 3), cmap = 'gray')
    # ax4.set_title(f"4 : {fold}", fontsize = "x-large")
    # ax4.grid(False)
    # ax5.imshow(np.rot90(region_pred[0, :, :, 5], 3), cmap = 'gray')
    # ax5.set_title(f"5 : {fold}", fontsize = "x-large")
    # ax5.grid(False)
    # ax6.imshow(np.rot90(region_pred[0, :, :, 6], 3), cmap = 'gray')
    # ax6.set_title(f"6 : {fold}", fontsize = "x-large")
    # ax6.grid(False)
    # ax7.imshow(np.rot90(region_pred[0, :, :, 7], 3), cmap = 'gray')
    # ax7.set_title(f"7 : {fold}", fontsize = "x-large")
    # ax7.grid(False)
    # ax8.imshow(np.rot90(region_pred[0, :, :, 8], 3), cmap = 'gray')
    # ax8.set_title(f"8 : {fold}", fontsize = "x-large")
    # ax8.grid(False)
    # ax9.imshow(np.rot90(region_pred[0, :, :, 9], 3), cmap = 'gray')
    # ax9.set_title(f"9 : {fold}", fontsize = "x-large")
    # ax9.grid(False)
    # # plt.savefig(f'H:/TOPIC/Unet_liver_segmentation/fold/segmentation_photo/fold{fold+1}/img_{fold}_seg_{j}.png')
    # plt.show()
    # plt.close()

    colors = [
        [70, 145, 200],   # 1
        [180, 210, 110],  # 2
        [100, 170, 130],  # 3
        [130, 110, 160],  # 4A
        [130, 110, 160],  # 4B
        [170, 170, 80],   # 5
        [170, 60, 60],    # 6
        [180, 110, 110],  # 7
        [240, 240, 170]   # 8
    ]
    for s in range(region_pred.shape[0]):
        result_image = np.zeros((region_pred.shape[1], region_pred.shape[2], 3), dtype=np.uint8)  # 預測的
        merged_image = np.zeros((region_pred.shape[1], region_pred.shape[2], 3), dtype=np.uint8)
        ori_image = train_data[s, :, :]
        mask = prediction_liver[s, :, :, 0]
        # tumor = train_label[s, :, :, 0]
        for j in range(region_pred.shape[3]):

            print("-----------------Segmentation-----------------")
            pre = region_pred[s, :, :, j]
            
            # pre[pre>=0.7] = 1
            # pre[pre<0.7] = 0

            # ori = region_pred[s, :, :, j]
                
            pre = np.expand_dims(pre, axis=-1)
            # ori = np.expand_dims(ori, axis=-1)
            print("type(pre) : ", type(pre))
            print("np.unique(pre) : ", np.unique(pre))
            
            if (j != 0):

                # 將預測出來的區域做上色
                background_color = [0, 0, 0]
                ori_background_color = [0, 0, 0]
                result_image = np.where(pre == 1, colors[j-1], background_color)
                merged_image[result_image != background_color] = result_image[result_image != background_color]
            
            # 建立與 merged_image 相同大小的全黑圖像
            bitwise = np.zeros_like(merged_image)

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize = ((20, 20)))

        ax0.imshow(ori_image, cmap = 'gray')
        ax0.set_title(f"mask_tumor : {fold}", fontsize = "x-large")
        ax0.grid(False)
        ax1.imshow(mask, cmap = 'gray')
        ax1.set_title(f"mask : {fold}", fontsize = "x-large")
        ax1.grid(False)
        ax2.imshow(merged_image, cmap = 'gray')
        ax2.set_title(f"predict : {fold}", fontsize = "x-large")
        ax2.grid(False)
        ax3.imshow(bitwise, cmap = 'gray')
        ax3.set_title(f"bitwise : {fold}", fontsize = "x-large")
        ax3.grid(False)
        # plt.savefig(f'H:/TOPIC/Unet_liver_segmentation/fold/segmentation_photo/fold{fold+1}/img_{fold}_seg_{j}.png')
        plt.show()
        plt.close()