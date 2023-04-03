"""
Program to use CRC-ResNet50 model for classifying images 
Package requirements:
    a) python = 3.9 
    b) numpy
    c) pandas
    d) tensorflow=2.11.0
    e) keras = 2.11.0
    f) scikit-learn
    g) scipy
    
Input: 
    a) generated-model.h5 file, 
    b) Folder named 'images' with images to be classified
Output: Prediction file
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import asarray
from os import listdir
from scipy import stats
from sklearn import preprocessing
from numpy import savetxt

path = "------path of the extracted folder------"

# Functions to load image data, model and perform classification task
def load_dataset(folder):
 	photos = list()
 	# enumerate files in the directory
 	for filename in listdir(folder):
         #load image
         photo = load_img(folder + filename, target_size=(240,240))
         photo = img_to_array(photo, dtype='uint8')
         photos.append(photo)
		
 	X = asarray(photos, dtype='uint8')
 	
 	return X

def cnn_prediction(path):
    folder= path+'/images/'
    X = load_dataset(folder)
    print("Images loaded")
    model=load_model("generated_model.h5")
    print("Model loaded")
    cnn_prediction=(model.predict(X))
    
    return cnn_prediction

def max_in_array(x):
  xx = x
  b=(xx == xx.max(axis=1, keepdims=1)).astype(int)
  return b

labels = np.array(['esophagitis', 'normal', 'polyps', 'ulcer'])
lb = preprocessing.LabelBinarizer()
y_labels = lb.fit_transform(labels)

#Prediction
yhats2 = cnn_prediction(path)
predict = max_in_array(yhats2)
y_prediction = lb.inverse_transform(predict)
print ("Predicted GI disease: ",y_prediction)

#Saving predictions
DF = pd.DataFrame(y_prediction)
DF.to_csv(path+"prediction.csv")
print ("Prediction saved")
