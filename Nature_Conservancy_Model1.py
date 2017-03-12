
# coding: utf-8

# # Detecting Fish - Model 1

# The goal of this model is to accurately detect which type of fish is in each image. The data is from a Kaggle competition, "The Nature Conservancy Fisheries Monitoring": https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring.
# 
# This model is slightly different than model 2. Here we will use a wider and deeper neural network, and an 80-20 train-test split, instead of Kfold with 5 splits.

# In[1]:

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD


# In[2]:

# The directories where the images are located
train_dir = 'train/'
test_dir = 'test_stg1/'
fish_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# In[3]:

# The size of the images we will use to train our model
rows = 64
cols = 64
channels = 3


# In[4]:

def get_images(fish):
    """Load files from train folder"""
    fish_dir = train_dir+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (cols, rows), interpolation=cv2.INTER_CUBIC)
    return im


files = []
y_all = []

for fish in fish_classes:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
        
y_all = np.array(y_all)


# In[5]:

# Process all of the training images
X_all = np.ndarray((len(files), rows, cols, channels), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(train_dir+im)
    if i%100 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)


# In[6]:

# View a fish from each class
uniq = np.unique(y_all, return_index=True)
for f, i in zip(uniq[0], uniq[1]):
    plt.imshow(X_all[i])
    plt.title(f)
    plt.show()


# In[7]:

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)


# In[8]:

# Normalize the pixel values
X_all = X_all.astype('float32')
X_all = X_all / 255


# In[13]:

x_train, x_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                      test_size=0.2, 
                                                      random_state=2)


# Note: This model takes quite a long time to train on my laptop. If I had greater processing power, it would have been nice to build a larger network similar to the ones in this research paper: https://arxiv.org/pdf/1409.1556.pdf
# 
# I used only one max pooling layer to reduce the amount of information that is lose within the image.
# 
# Dropout isn't used because the model really struggled to learn when it was included. By transforming the images with ImageDataGenerator, this prevented overfitting from occuring. 

# In[14]:

def model():
    # create model
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(rows, cols, channels), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(fish_classes), activation='softmax'))
    # Compile model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])
    return model


# In[15]:

# Transform the images to prevent overfitting.
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
    ) 

datagen.fit(x_train)

model = model()

# Use EarlyStopping to stop the training when the learning plateaus.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

model.fit_generator(datagen.flow(x_train, y_train, 
                                 batch_size = 24),
                    samples_per_epoch = len(x_train), 
                    nb_epoch = 250, 
                    verbose = 2,
                    validation_data = (x_valid, y_valid),
                    callbacks = [early_stopping])


# In[16]:

# Double check the validation predictions
preds = model.predict(x_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))


# In[17]:

# Load the testing images to be submitted to Kaggle.
test_files = [im for im in os.listdir(test_dir)]
test = np.ndarray((len(test_files), rows, cols, channels), dtype=np.uint8)

# Process the images
for i, im in enumerate(test_files): 
    test[i] = read_image(test_dir+im)
    
# Normalize
test = test.astype('float32')
test = test/255


# In[18]:

test_preds = model.predict(test, verbose=1)


# In[19]:

submission = pd.DataFrame(test_preds, columns=fish_classes)
submission.insert(0, 'image', test_files)
submission.to_csv('submission.csv', index=False, header=True)
submission.head(10)

