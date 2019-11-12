import tensorflow.compat.v1 as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, Dense, Dropout
from keras.layers.core import Activation
from keras.optimizers import Adam
import numpy as np


def FashionNet_dense(image_rows, image_cols, img_channels=1, optimizer=Adam(lr=3e-4)):
    inputs = Input((image_rows, image_cols, img_channels))

    flatt = Flatten()(inputs)

    dens1 = Dense(128, activation='relu')(flatt)
    drop1 = Dropout(0.3)(dens1)

    dens2 = Dense(256, activation='relu')(drop1)
    drop2 = Dropout(0.4)(dens2)

    dens3 = Dense(512, activation='relu')(drop2)
    drop3 = Dropout(0.5)(dens3)

    dens4 = Dense(64, activation='relu')(drop3)
    drop4 = Dropout(0.5)(dens4)

    dens5 = Dense(10, activation='softmax')(drop4)

    model = Model(inputs=[inputs], outputs=[dens5])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def FashionNet_conv1d(image_rows, image_cols, img_channels=1, optimizer=Adam(lr=3e-4)):
    inputs = Input((image_rows, image_cols, img_channels))
 
    conv1 = Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, (1, 3), activation='relu')(conv1)
    conv1 = Conv2D(64, (1, 1), activation='relu')(conv1)
     
    pool1 = MaxPooling2D(pool_size=(1, 2))(conv1)
    drop  = Dropout(0.5)(pool1)
 
    conv2 = Conv2D(32, (1, 1), activation='relu')(drop)
    conv2 = Conv2D(32, (1, 3), activation='relu')(conv2)
    conv2 = Conv2D(128, (1, 1), activation='relu')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(1, 2))(conv2)
    drop0 = Dropout(0.5)(pool2)
 
    flatt = Flatten()(drop0)
 
    dens2 = Dense(64, activation='relu')(flatt)
    drop2 = Dropout(0.4)(dens2)
 
    dens4 = Dense(10, activation='softmax')(drop2)
 
    model = Model(inputs=[inputs], outputs=[dens4])
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
