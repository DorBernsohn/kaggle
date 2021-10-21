import json
import tensorflow as tf
from keras.layers import (Input,
                          Dense,
                          Conv2D,
                          Dropout,
                          Flatten,
                          Activation,
                          MaxPooling2D,
                          BatchNormalization)

from layers import ArcFace, CosFace, SphereFace

with open('config.json') as f:
  config = json.load(f)

weight_decay = config["weight_decay"]
kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), 
                   padding='same', 
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x

def base_block(num_features, input_shape):
    input = Input(shape=input_shape)
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(num_features, 
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    return input, x, y

def vgg8(num_features, input_shape=(28, 28, 1)):
    input, x, _ = base_block(num_features, input_shape)
    output = Dense(10, activation='softmax', kernel_regularizer=kernel_regularizer)(x)

    return tf.keras.models.Model(input, output)

def vgg8_arcface(num_features, input_shape=(28, 28, 1)):
    input, x, y = base_block(num_features, input_shape)
    output = ArcFace(10, regularizer=kernel_regularizer)([x, y])

    return tf.keras.models.Model([input, y], output)

def vgg8_cosface(num_features, input_shape=(28, 28, 1)):    
    input, x, y = base_block(num_features, input_shape)
    output = CosFace(10, regularizer=kernel_regularizer)([x, y])

    return tf.keras.models.Model([input, y], output)

def vgg8_sphereface(num_features, input_shape=(28, 28, 1)):
    input, x, y = base_block(num_features, input_shape)
    output = SphereFace(10, regularizer=kernel_regularizer)([x, y])

    return tf.keras.models.Model([input, y], output)