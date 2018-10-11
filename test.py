#D:/kaggle/表情识别/test.py
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# get the data
filname = r'D:/kaggle/表情识别/test.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names=['id','feature']
df=pd.read_csv(r'D:/kaggle/表情识别/test.csv',names=names, na_filter=False)
im=df['feature']
df.head(10)

def getData(filname):
    # images are 48x48
    # N = 28709
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X = np.array(X) / 255.0
    return X

X = getData(filname)

N, D = X.shape
X = X.reshape(N, 48, 48, 1)

from keras.models import load_model

model = load_model('D:/kaggle/表情识别/my_model.h5')
Y = model.predict(X)

ans = []
rows,columns=Y.shape
for i in range(rows):
    tmp=0
    for j in range(columns):
        if Y[i,j]>Y[i,tmp]:
            tmp=j
    ans.append(tmp)

np.savetxt('D:/kaggle/表情识别/answer.csv',ans,fmt='%lf')