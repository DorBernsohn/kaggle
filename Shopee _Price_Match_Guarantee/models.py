import json
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Flatten

from layers import ArcFace, CosFace, SphereFace
from utils import CosineAnnealingScheduler

with open('config.json') as f:
  config = json.load(f)
  
class VisionBertSoftmaxModel(tf.keras.Model):
    """create a model that concat BERT embeddings and other vision model (VGG, ResNet..)
    Args:
        tf (tf.keras.Model): tf.keras.Model
    """
    def __init__(self, seq_len: int = 100,
                       text_model_name: str = 'bert-base-uncased',
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False),
                       num_labels: int = 1) -> None:

        super(VisionBertSoftmaxModel, self).__init__()
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        
        self.vision_model = vision_model
        self.vision_model.trainable = False

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.concat = Concatenate(axis=1)

        self.global_dense1 = Dense(2048, activation='relu')
        self.global_dense3 = Dense(num_labels, activation='softmax')
        self.dense_text1 = Dense(768, activation='relu')
        self.img_dense1 = Dense(2048, activation='relu')
         
    def call(self, inputs: list) -> tf.keras.layers:
        
        text_inputs = inputs[:3]
        img_inputs = inputs[-1]
        text = self.text_model_layer(text_inputs)[1]

        text = self.dense_text1(text)
        
        img = img_inputs
        img = self.img_dense1(img)

        concat = self.concat([text, img])
        dropout = self.dropout(concat)
        concat = self.global_dense1(dropout)
        return self.global_dense3(concat)

class VisionBertSoftmaxClassifier:
    """build the model and run training
    """
    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100,
                       num_labels: int = 1) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        self.num_labels = num_labels
        
    def encode(self, texts: list) -> Tuple[np.array, np.array, np.array]:

        encoded = self.tokenizer.batch_encode_plus(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )
        return np.array(encoded["input_ids"], dtype="int32"), np.array(encoded["attention_mask"], dtype="int32"), np.array(encoded["token_type_ids"], dtype="int32")
    
    def build(self, vision_model: tf.keras.applications) -> tf.keras.Model:
        
        METRICS = [tf.keras.metrics.CategoricalAccuracy()]

        model = VisionBertSoftmaxModel(seq_len=self.seq_len, vision_model=vision_model, num_labels = self.num_labels)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(config["lr"]), metrics=METRICS)

        return model

    def train(self, data: dict, 
                    vision_model: tf.keras.applications,
                    validation_split: float = 0.2) -> tf.keras.callbacks.History:

        self.model = self.build(vision_model)
        
        train_index = int(data['text'].shape[0]*0.8)

        train_text = self.encode(data['text'][:train_index])
        test_text = self.encode(data['text'][train_index:])
        
        train_image = data['image'][:train_index]
        test_image = data['image'][train_index:]
        
        train_labels = np.asarray(data['label'][:train_index])
        test_labels = np.asarray(data['label'][train_index:])
        
        self.validation_data = ([test_text[0], test_text[1], test_text[2], test_image], test_labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        cosine_learing_rate_callback = CosineAnnealingScheduler(T_max=100,
                                                        eta_max=1e-2,
                                                        eta_min=1e-4,
                                                        verbose=1)
        
        self.history = self.model.fit([train_text[0], train_text[1], train_text[2], train_image],
                                 train_labels,
                                 validation_data=self.validation_data,
                                 batch_size=config["batch_size"],
                                 callbacks = [callback, cosine_learing_rate_callback],
                                 epochs=config["ephocs"])

        return self.history

class VisionBertArcFaceModel(tf.keras.Model):
    """create a model that concat BERT embeddings and other vision model (VGG, ResNet..)
    Args:
        tf (tf.keras.Model): tf.keras.Model
    """
    def __init__(self, seq_len: int = 100,
                       text_model_name: str = 'bert-base-uncased',
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False),
                       num_labels: int = 1) -> None:

        super(VisionBertArcFaceModel, self).__init__()
        kernel_regularizer = tf.keras.regularizers.l2(config["regularizer_penalty"])
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        
        self.vision_model = vision_model
        self.vision_model.trainable = False

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.concat = Concatenate(axis=1)

        self.global_dense1 = Dense(2048, activation='relu')
        self.arcface = ArcFace(num_labels, regularizer=kernel_regularizer)
        self.dense_text1 = Dense(768, activation='relu')
        self.img_dense1 = Dense(2048, activation='relu')
         
    def call(self, inputs: list) -> tf.keras.layers:
        text_inputs = inputs[0][:3]
        img_inputs = inputs[0][-1]
        text = self.text_model_layer(text_inputs)[1]

        text = self.dense_text1(text)
        
        img = img_inputs
        img = self.img_dense1(img)

        concat = self.concat([text, img])
        x = self.dropout(concat)
        x = self.global_dense1(x)
        x = self.arcface([x, inputs[1]])
        return x

class VisionBertArcFaceClassifier:
    """build the model and run training
    """
    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100,
                       num_labels: int = 1) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        self.num_labels = num_labels
        
    def encode(self, texts: list) -> Tuple[np.array, np.array, np.array]:

        encoded = self.tokenizer.batch_encode_plus(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )
        return np.array(encoded["input_ids"], dtype="int32"), np.array(encoded["attention_mask"], dtype="int32"), np.array(encoded["token_type_ids"], dtype="int32")
    
    def build(self, vision_model: tf.keras.applications):
        
        METRICS = [tf.keras.metrics.CategoricalAccuracy()]

        model = VisionBertArcFaceModel(seq_len=self.seq_len, vision_model=vision_model, num_labels = self.num_labels)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(config["lr"]), metrics=METRICS)

        return model

    def train(self, data: dict, 
                    vision_model: tf.keras.applications,
                    validation_split: float = 0.2) -> tf.keras.callbacks.History:

        self.model = self.build(vision_model)
        
        train_index = int(data['text'].shape[0]*0.8)

        train_text = self.encode(data['text'][:train_index])
        test_text = self.encode(data['text'][train_index:])
        
        train_image = data['image'][:train_index]
        test_image = data['image'][train_index:]
        
        train_labels = np.asarray(data['label'][:train_index])
        test_labels = np.asarray(data['label'][train_index:])
        
        self.validation_data = ([[test_text[0], test_text[1], test_text[2], test_image], test_labels], test_labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        cosine_learing_rate_callback = CosineAnnealingScheduler(T_max=100,
                                                        eta_max=1e-2,
                                                        eta_min=1e-4,
                                                        verbose=1)
        
        self.history = self.model.fit([[train_text[0], train_text[1], train_text[2], train_image], train_labels],
                                 train_labels,
                                 validation_data=self.validation_data,
                                 batch_size=config["batch_size"],
                                 callbacks = [callback, cosine_learing_rate_callback],
                                 epochs=config["ephocs"])

        return self.history

class VisionBertCosFaceModel(tf.keras.Model):
    """create a model that concat BERT embeddings and other vision model (VGG, ResNet..)
    Args:
        tf (tf.keras.Model): tf.keras.Model
    """
    def __init__(self, seq_len: int = 100,
                       text_model_name: str = 'bert-base-uncased',
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False),
                       num_labels: int = 1) -> None:

        super(VisionBertCosFaceModel, self).__init__()
        kernel_regularizer = tf.keras.regularizers.l2(config["regularizer_penalty"])
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        
        self.vision_model = vision_model
        self.vision_model.trainable = False

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.concat = Concatenate(axis=1)

        self.global_dense1 = Dense(2048, activation='relu')
        self.cosface = CosFace(num_labels, regularizer=kernel_regularizer)
        self.dense_text1 = Dense(768, activation='relu')
        self.img_dense1 = Dense(2048, activation='relu')
         
    def call(self, inputs: list) -> tf.keras.layers:
        text_inputs = inputs[0][:3]
        img_inputs = inputs[0][-1]
        text = self.text_model_layer(text_inputs)[1]

        text = self.dense_text1(text)
        
        img = img_inputs
        img = self.img_dense1(img)

        concat = self.concat([text, img])
        x = self.dropout(concat)
        x = self.global_dense1(x)
        x = self.cosface([x, inputs[1]])
        return x

class VisionBertCosFaceClassifier:
    """build the model and run training
    """
    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100,
                       num_labels: int = 1) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        self.num_labels = num_labels
        
    def encode(self, texts: list) -> Tuple[np.array, np.array, np.array]:

        encoded = self.tokenizer.batch_encode_plus(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )
        return np.array(encoded["input_ids"], dtype="int32"), np.array(encoded["attention_mask"], dtype="int32"), np.array(encoded["token_type_ids"], dtype="int32")
    
    def build(self, vision_model: tf.keras.applications):
        
        METRICS = [tf.keras.metrics.CategoricalAccuracy()]

        model = VisionBertCosFaceModel(seq_len=self.seq_len, vision_model=vision_model, num_labels = self.num_labels)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(config["lr"]), metrics=METRICS)

        return model

    def train(self, data: dict, 
                    vision_model: tf.keras.applications,
                    validation_split: float = 0.2) -> tf.keras.callbacks.History:

        self.model = self.build(vision_model)
        
        train_index = int(data['text'].shape[0]*0.8)

        train_text = self.encode(data['text'][:train_index])
        test_text = self.encode(data['text'][train_index:])
        
        train_image = data['image'][:train_index]
        test_image = data['image'][train_index:]
        
        train_labels = np.asarray(data['label'][:train_index])
        test_labels = np.asarray(data['label'][train_index:])
        
        self.validation_data = ([[test_text[0], test_text[1], test_text[2], test_image], test_labels], test_labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        cosine_learing_rate_callback = CosineAnnealingScheduler(T_max=100,
                                                        eta_max=1e-2,
                                                        eta_min=1e-4,
                                                        verbose=1)
        
        self.history = self.model.fit([[train_text[0], train_text[1], train_text[2], train_image], train_labels],
                                 train_labels,
                                 validation_data=self.validation_data,
                                 batch_size=config["batch_size"],
                                 callbacks = [callback, cosine_learing_rate_callback],
                                 epochs=config["ephocs"])

        return self.history

class VisionBertSphereFaceModel(tf.keras.Model):
    """create a model that concat BERT embeddings and other vision model (VGG, ResNet..)
    Args:
        tf (tf.keras.Model): tf.keras.Model
    """
    def __init__(self, seq_len: int = 100,
                       text_model_name: str = 'bert-base-uncased',
                       vision_model: tf.keras.applications = VGG19(weights="imagenet", include_top=False),
                       num_labels: int = 1) -> None:

        super(VisionBertSphereFaceModel, self).__init__()
        kernel_regularizer = tf.keras.regularizers.l2(config["regularizer_penalty"])
        self.text_model_layer = TFAutoModel.from_pretrained(text_model_name)
        self.text_model_layer.trainable = False
        
        self.vision_model = vision_model
        self.vision_model.trainable = False

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.concat = Concatenate(axis=1)

        self.global_dense1 = Dense(2048, activation='relu')
        self.sphereface = SphereFace(num_labels, regularizer=kernel_regularizer)
        self.dense_text1 = Dense(768, activation='relu')
        self.img_dense1 = Dense(2048, activation='relu')
         
    def call(self, inputs: list) -> tf.keras.layers:
        text_inputs = inputs[0][:3]
        img_inputs = inputs[0][-1]
        text = self.text_model_layer(text_inputs)[1]

        text = self.dense_text1(text)
        
        img = img_inputs
        img = self.img_dense1(img)

        concat = self.concat([text, img])
        x = self.dropout(concat)
        x = self.global_dense1(x)
        x = self.sphereface([x, inputs[1]])
        return x

class VisionBertSphereFaceClassifier:
    """build the model and run training
    """
    def __init__(self, text_model_name: str = 'bert-base-uncased', 
                       seq_len: int = 100,
                       num_labels: int = 1) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.seq_len = seq_len
        self.num_labels = num_labels
        
    def encode(self, texts: list) -> Tuple[np.array, np.array, np.array]:

        encoded = self.tokenizer.batch_encode_plus(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )
        return np.array(encoded["input_ids"], dtype="int32"), np.array(encoded["attention_mask"], dtype="int32"), np.array(encoded["token_type_ids"], dtype="int32")
    
    def build(self, vision_model: tf.keras.applications):
        
        METRICS = [tf.keras.metrics.CategoricalAccuracy()]

        model = VisionBertSphereFaceModel(seq_len=self.seq_len, vision_model=vision_model, num_labels = self.num_labels)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(config["lr"]), metrics=METRICS)

        return model

    def train(self, data: dict, 
                    vision_model: tf.keras.applications,
                    validation_split: float = 0.2) -> tf.keras.callbacks.History:

        self.model = self.build(vision_model)
        
        train_index = int(data['text'].shape[0]*0.8)

        train_text = self.encode(data['text'][:train_index])
        test_text = self.encode(data['text'][train_index:])
        
        train_image = data['image'][:train_index]
        test_image = data['image'][train_index:]
        
        train_labels = np.asarray(data['label'][:train_index])
        test_labels = np.asarray(data['label'][train_index:])
        
        self.validation_data = ([[test_text[0], test_text[1], test_text[2], test_image], test_labels], test_labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        cosine_learing_rate_callback = CosineAnnealingScheduler(T_max=100,
                                                        eta_max=1e-2,
                                                        eta_min=1e-4,
                                                        verbose=1)
        
        self.history = self.model.fit([[train_text[0], train_text[1], train_text[2], train_image], train_labels],
                                 train_labels,
                                 validation_data=self.validation_data,
                                 batch_size=config["batch_size"],
                                 callbacks = [callback, cosine_learing_rate_callback],
                                 epochs=config["ephocs"])

        return self.history

models = {"Softmax": VisionBertSoftmaxClassifier,
          "ArcFace": VisionBertArcFaceClassifier,
          "CosFace": VisionBertCosFaceClassifier,
          "SphereFace": VisionBertSphereFaceClassifier}