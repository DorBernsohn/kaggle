 # * @author  Dor Bernsohn

import transformers
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import TFXLNetModel, TFXLNetForSequenceClassification, XLNetTokenizer

def create_tfds_dataset(texts, lables):
    """create tensorflow dataset

    Args:
        texts ([list]): [list of strings/pandas column]
        lables ([list]): [list of strings/pandas column]

    Returns:
        [tfds]: [tensorflow dataset object]
    """    
    ds = tf.data.Dataset.from_tensor_slices((texts, texts))
    return ds

class BERTModel():
    """create a BERT inputs and build a model
    """        
    def __init__(self, ds_train, ds_test, max_length=512, batch_size=6, learning_rate = 2e-5):
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.ds_train = ds_train
        self.ds_test = ds_test
        
    def convert_example_to_feature(self, text):
        """combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length

        Args:
            text (string): text

        Returns:
            bert input: bert input for further processing
        """        
    
        return self.tokenizer.encode_plus(text,
                        add_special_tokens = True, # add [CLS], [SEP]
                        max_length = self.max_length, # max length of the text that can go to BERT
                        pad_to_max_length = True, # add [PAD] tokens
                        return_attention_mask = True, # add attention mask to not focus on pad tokens
                        truncation=True
                    )
    
    def map_example_to_dict(self, input_ids, attention_masks, token_type_ids, label):
        """map to the expected input to TFBertForSequenceClassification

        Args:
            input_ids (list): list of inputs ids
            attention_masks (list): list of attention masks
            token_type_ids (list): list of token type ids
            label (list): list of lables

        Returns:
            dictionary: dictionary of {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        """        
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks,
        }, label

    def encode_examples(self, ds, limit=-1):
        """prepare list, so that we can build up final TensorFlow dataset from slices

        Args:
            ds (tensorflow dataset): a tensorflow dataset of text and lables
            limit (int, optional): how many samples to take. Defaults to 10.

        Returns:
            tensorflow dataset object: a tensorflow dataset of bert inputs and lables
        """        
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []

        if (limit > 0):
            ds = ds.take(limit)
            
        for review, label in tqdm(tfds.as_numpy(ds)):

            bert_input = self.convert_example_to_feature(review.decode())
        
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(self.map_example_to_dict)
    
    def process_examples(self):
        """encode the dataset
        """        
        self.ds_train_encoded = self.encode_examples(self.ds_train).shuffle(10000).batch(self.batch_size)

        self.ds_test_encoded = self.encode_examples(self.ds_test).batch(self.batch_size)

    @staticmethod
    def build_model(learning_rate, epsilon=1e-08):
        """build the BERT model

        Args:
            learning_rate (float): the learning rate for the Adam optimizer
            epsilon (float, optional): the epsilon for the Adam optimizer. Defaults to 1e-08.

        Returns:
            tensorflow model object: the model object
        """        
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        print(model.summary())
        return model

    def training(self, number_of_epochs):
        model = self.build_model(learning_rate=self.learning_rate)
        my_callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min', baseline=None, restore_best_weights=True)
        ]
        bert_history = model.fit(
                         self.ds_train_encoded, 
                         validation_data=self.ds_test_encoded,
                         epochs=number_of_epochs,
                         callbacks=my_callbacks)

        return bert_history

class XLNetModel():
    """create a XLNet inputs and build a model
    """        
    def __init__(self, ds_train, ds_test, max_length=512, batch_size=6, learning_rate=2e-5):
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        self.ds_train = ds_train
        self.ds_test = ds_test
        
    def convert_example_to_feature(self, text):
        """combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length

        Args:
            text (string): text

        Returns:
            XLNet input: XLNet input for further processing
        """        
    
        return self.tokenizer.encode_plus(text,
                        max_length = self.max_length, # max length of the text that can go to BERT
                        pad_to_max_length = True, # add [PAD] tokens
                        truncation=True
                    )
    
    def map_example_to_dict(self, input_ids, attention_masks, token_type_ids, label):
        """map to the expected input to TFXLNetForSequenceClassification

        Args:
            input_ids (list): list of inputs ids
            attention_masks (list): list of attention masks
            token_type_ids (list): list of token type ids
            label (list): list of lables

        Returns:
            dictionary: dictionary of {"input_ids": input_ids, "attention_mask": attention_mask}
        """        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }, label

    def encode_examples(self, ds, limit=-1):
        """prepare list, so that we can build up final TensorFlow dataset from slices

        Args:
            ds (tensorflow dataset): a tensorflow dataset of text and lables
            limit (int, optional): how many samples to take. Defaults to 10.

        Returns:
            tensorflow dataset object: a tensorflow dataset of bert inputs and lables
        """        
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        attention_mask_list = []
        label_list = []

        if (limit > 0):
            ds = ds.take(limit)
            
        for review, label in tqdm(tfds.as_numpy(ds)):

            xlnet_input = self.convert_example_to_feature(review.decode())
        
            input_ids_list.append(xlnet_input['input_ids'])
            attention_mask_list.append(xlnet_input['attention_mask'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, label_list)).map(self.map_example_to_dict)
    
    def process_examples(self):
        """encode the dataset
        """        
        self.ds_train_encoded = self.encode_examples(self.ds_train).shuffle(10000).batch(self.batch_size)

        self.ds_test_encoded = self.encode_examples(self.ds_test).batch(self.batch_size)

    @staticmethod
    def build_model(learning_rate, epsilon=1e-08):
        """build the XLNet model

        Args:
            learning_rate (float): the learning rate for the Adam optimizer
            epsilon (float, optional): the epsilon for the Adam optimizer. Defaults to 1e-08.

        Returns:
            tensorflow model object: the model object
        """        
        model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        print(model.summary())
        return model

    def training(self, number_of_epochs):
        model = self.build_model(learning_rate=self.learning_rate)
        my_callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min', baseline=None, restore_best_weights=True)
        ]
        xlnet_history = model.fit(
                         self.ds_train_encoded, 
                         validation_data=self.ds_test_encoded,
                         epochs=number_of_epochs,
                         callbacks=my_callbacks)
            
        return xlnet_history