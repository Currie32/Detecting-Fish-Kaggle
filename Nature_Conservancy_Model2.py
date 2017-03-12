
# coding: utf-8

# # Detecting Fish - Method 2

# The goal of this model is to accurately detect which type of fish (if there is one) is in each image. The data is from a Kaggle competition, "The Nature Conservancy Fisheries Monitoring": https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring.
# 
# This model is slightly different than model 1. Here we will use a narrower and shallower neural network, and Kfold with 5 splits, instead of an 80-20 train-test split.

# In[14]:

import numpy as np
np.random.seed(2)

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import glob
import cv2
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adagrad, Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split


# In[6]:

# The directories where the images are located
train_dir = 'train/'
test_dir = 'test_stg1/'
fish_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# In[7]:

# The size of the images we will use to train our model
rows = 64
cols = 64
channels = 3


# In[80]:

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


# In[9]:

# Process all of the training images
X_all = np.ndarray((len(files), rows, cols, channels), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(train_dir+im)
    if i%100 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)


# In[12]:

# View a fish from each class
uniq = np.unique(y_all, return_index=True)
for f, i in zip(uniq[0], uniq[1]):
    plt.imshow(X_all[i])
    plt.title(f)
    plt.show()


# In[81]:

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)


# In[16]:

# Normalize the pixel values
X_all = X_all.astype('float32')
X_all = X_all / 255


# In[82]:

def cnn_model():
    model = Sequential()
    model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(rows, cols, channels), activation='relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(fish_classes), activation='softmax'))

    # The learning rate in this model is higher than in model 1, 0.002 vs 0.001.
    # The model is designed to train faster than model 1.
    sgd = SGD(lr=0.002, decay=5e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

    return model


# In[83]:

# Parameters
batch_size = 32
nb_epoch = 250
random_state = 2
nFolds = 5 

# Transform the images to prevent overfitting.
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    )

# Split the data into the number of folds (nFolds)
kf = KFold(len(X_all), n_folds=nFolds, shuffle=True, random_state=random_state)
num_fold = 0 # Keeps track of the current fold
sum_score = 0 # Keeps track of the total score/log loss
models = [] # Contains the model for each of the folds

for train_index, test_index in kf:
    model = cnn_model()
    x_train = X_all[train_index]
    y_train = y_all[train_index]
    x_valid = X_all[test_index]
    y_valid = y_all[test_index]
    datagen.fit(x_train)

    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nFolds))
    print('Split train:', len(x_train), len(y_train))
    print('Split valid:', len(x_valid), len(y_valid))
 
    # Use EarlyStopping to stop the training when the learning plateaus.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    
    # Transform data and train the model
    model.fit_generator(datagen.flow(x_train, y_train, 
                                     batch_size = batch_size),
                        samples_per_epoch=len(x_train), 
                        nb_epoch = nb_epoch, 
                        verbose = 2,
                        validation_data = (x_valid, y_valid),
                        callbacks = [early_stopping])

    # Make the predictions
    predictions_valid = model.predict(x_valid, verbose=1)
    
    # Record and sum score/log loss
    score = log_loss(y_valid, predictions_valid)
    print('Score log_loss: ', score)
    sum_score += score

    # Store valid predictions
    for i in range(len(test_index)):
        y_all[test_index[i]] = predictions_valid[i]

    models.append(model)

# Average the validation scores from all of the folds.
score = sum_score/nFolds
print("Log_loss avg: ", score)
info_string = 'loss_' + str(score) + '_folds_' + str(nFolds) + '_ep_' + str(nb_epoch)


# In[74]:

# Load the testing images to be submitted to Kaggle.
test_files = [im for im in os.listdir(test_dir)]
test = np.ndarray((len(test_files), rows, cols, channels), dtype=np.uint8)

# Process the images
for i, im in enumerate(test_files): 
    test[i] = read_image(test_dir+im)
    
# Normalize
test = test.astype('float32')
test = test/255


# In[84]:

#def run_cross_validation_process_test(info_string, models):
batch_size = 32
num_fold = 0
predictions_list = []
test_id = []
nfolds = len(models)

for i in range(nfolds):
    # Make predictions with each model from each fold.
    model = models[i]
    num_fold += 1
    print('Start KFold number {} from {}'.format(num_fold, nfolds))
    test_predictions = model.predict(test, batch_size=batch_size, verbose=2)
    # Add each set of predictions to predictions_list
    predictions_list.append(test_predictions)
    
def merge_several_folds_mean(data, nfolds):
    '''Averages the predictions made, and converts the averaged predictions into a list'''
    predictions_made = np.array(data[0])
    for i in range(1, nfolds):
        predictions_made += np.array(data[i])
    predictions_made /= nfolds
    return predictions_made.tolist()

submission_preds = merge_several_folds_mean(predictions_list, nfolds)


# In[85]:

#Create submission
submission = pd.DataFrame(submission_preds, columns=fish_classes)
submission.insert(0, 'image', test_files)
file_name = 'submission_' + info_string + '.csv'
submission.to_csv(file_name, index=False)


# In[86]:

submission.head(10)

