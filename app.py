#from keras.utils import normalize
from flask import Flask, render_template, request
from keras.utils import normalize
from keras.models import load_model
from skimage.transform import resize
from keras.callbacks import Callback
import shutil
import gzip
from keras.preprocessing.image import ImageDataGenerator
import math
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping  # early stoppping
from model import *
from random import shuffle
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nib
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import set_session  # 分配GPU會用到
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import BatchNormalization as bn
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow import keras
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import tensorflow.keras as keras
import tensorflow as tf
# from Warning import *


app = Flask(__name__)

# model = load_model(f'H:/TOPIC/Unet_liver_segmentation/h5_file/lits_model_1.h5', compile=False)
# Create file from database
with open('H:/TOPIC/Unet_liver_segmentation/h5_file/lits_model_1.h5', "wb") as filehandler:
    test = Stock.query.filter_by(symbol=str(
        stock.symbol) + 'H:/TOPIC/Unet_liver_segmentation/h5_file/lits_model_1.h5').first()
    filehandler.write(test.data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_file', methods=['POST'])
def process_file():

    file = request.files['file']
    file.save('uploaded_file.nii')

    nii_file = nib.load('uploaded_file.nii')

    prediction_liver = model.predict(nii_file)

    prediction_liver[prediction_liver >= 0.9] = 1
    prediction_liver[prediction_liver < 0.9] = 0

    result = np.shape(prediction_liver)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run()
