 # @author  DBernsohn

import transformers
from tqdm import tqdm
import tensorflow as tf

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import TFXLNetModel, TFXLNetForSequenceClassification, XLNetTokenizer

class BertInputs():
    """create a BERT inputs
    """        
    def __init__(self, texts, lables, max_length=512, batch_size=6, bert_model_name='bert-base-uncased'):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
        self.texts = texts
        self.lables = lables
        
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
                        return_token_type_ids = True,
                        truncation=True
                    )
    @staticmethod
    def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
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

    def encode_examples(self, texts, lables, limit=-1):
        """prepare list, so that we can build up final TensorFlow dataset from slices

        Args:
            texts (list): list of texts
            lables (list): list of lables
            limit (int, optional): how many samples to take. Defaults to 10.

        Returns:
            tensorflow dataset object: a tensorflow dataset of bert inputs and lables
        """        
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        label_list = []
            
        for text, label in tqdm(zip(texts, lables)):

            bert_input = self.convert_example_to_feature(text)
        
            input_ids_list.append(bert_input['input_ids'])
            token_type_ids_list.append(bert_input['token_type_ids'])
            attention_mask_list.append(bert_input['attention_mask'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(self.map_example_to_dict)
    
    def process_examples(self, train=None):
        """encode the dataset

        Returns:
            tensorflow dataset objext: the text encode to BERT inputs as a tensorflow object
        """        
        if train:
            ds_train_encoded = self.encode_examples(self.texts, self.lables).shuffle(10000).batch(self.batch_size)
            return ds_train_encoded
        else:
            ds_test_encoded = self.encode_examples(self.texts, self.lables).batch(self.batch_size)
            return ds_test_encoded

class XLNetInputs():
    """create a XLNet inputs and build a model
    """        
    def __init__(self, texts, lables, max_length=512, batch_size=6, xlnet_model_name='xlnet-base-cased'):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = XLNetTokenizer.from_pretrained(xlnet_model_name, do_lower_case=True)
        self.texts = texts
        self.lables = lables
        
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
    @staticmethod
    def map_example_to_dict(input_ids, attention_masks, label):
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

    def encode_examples(self, texts, lables, limit=-1):
        """prepare list, so that we can build up final TensorFlow dataset from slices

        Args:
            texts (list): list of texts
            lables (list): list of lables
            limit (int, optional): how many samples to take. Defaults to 10.

        Returns:
            tensorflow dataset object: a tensorflow dataset of bert inputs and lables
        """        
        # prepare list, so that we can build up final TensorFlow dataset from slices.
        input_ids_list = []
        attention_mask_list = []
        label_list = []
            
        for review, label in tqdm(zip(texts, lables)):

            xlnet_input = self.convert_example_to_feature(review.decode())
        
            input_ids_list.append(xlnet_input['input_ids'])
            attention_mask_list.append(xlnet_input['attention_mask'])
            label_list.append([label])

        return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, label_list)).map(self.map_example_to_dict)
    
    def process_examples(self, train=None):
        """encode the dataset

        Returns:
            tensorflow dataset objext: the text encode to BERT inputs as a tensorflow object
        """        
        if train:
            ds_train_encoded = self.encode_examples(self.texts, self.lables).shuffle(10000).batch(self.batch_size)
            return ds_train_encoded
        else:
            ds_test_encoded = self.encode_examples(self.texts, self.lables).batch(self.batch_size)
            return ds_test_encoded