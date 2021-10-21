import tensorflow as tf
from typing import Tuple
from abc import ABC, abstractmethod

class LayerFaceClass(ABC, tf.keras.layers.Layer):
    def __init__(self, n_classes: int = 10, s: float = 30.0, m: float = 0.50, regularizer: tf.keras.regularizers = None, **kwargs) -> None:
        super(LayerFaceClass, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self) -> dict:
      config = super().get_config().copy()
    
      config.update({
          'n_classes': self.n_classes,
          's': self.s,
          'm': self.m,
          'regularizer': self.regularizer
      })
      return config

    def build(self, input_shape: tuple):
        super(LayerFaceClass, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer=tf.keras.initializers.GlorotUniform(seed=42),
                                 trainable=True,
                                 regularizer=self.regularizer)
                                 
    def compute_output_shape(self, input_shape: tuple) -> Tuple[None, int]:
        return (None, self.n_classes)

    @abstractmethod
    def call(self, inputs):
        pass

class ArcFace(LayerFaceClass):
    def call(self, inputs):
        x, y = inputs
        
        x = tf.nn.l2_normalize(x, axis=1) # normalize feature
        W = tf.nn.l2_normalize(self.W, axis=0) # normalize weights

        logits = x @ W # dot product

        theta = tf.acos(tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())) # clip logits to prevent zero division when backward

        target_logits = tf.cos(theta + self.m)

        logits = logits * (1 - y) + target_logits * y

        logits *= self.s # feature re-scale

        out = tf.nn.softmax(logits)

        return out



class SphereFace(LayerFaceClass):
    def call(self, inputs):
        x, y = inputs

        x = tf.nn.l2_normalize(x, axis=1) # normalize feature
        W = tf.nn.l2_normalize(self.W, axis=0) # normalize weights

        logits = x @ W # dot product

        theta = tf.acos(tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())) # clip logits to prevent zero division when backward

        target_logits = tf.cos(self.m * theta)

        logits = logits * (1 - y) + target_logits * y

        logits *= self.s # feature re-scale

        out = tf.nn.softmax(logits)

        return out

class CosFace(LayerFaceClass):
    def call(self, inputs):
        x, y = inputs

        x = tf.nn.l2_normalize(x, axis=1) # normalize feature
        W = tf.nn.l2_normalize(self.W, axis=0) # normalize weights

        logits = x @ W # dot product

        target_logits = logits - self.m # add margin

        logits = logits * (1 - y) + target_logits * y

        logits *= self.s # feature re-scale

        out = tf.nn.softmax(logits)

        return out
