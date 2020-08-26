 # @author  DBernsohn

import re
import nltk
import emoji
import string
import warnings
import tensorflow as tf
import gensim.downloader as api
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from transformers import pipeline
from googletrans import Translator

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')  
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='transformers')

def clean_text(text, remove_emojis=True, remove_numbers=True, remove_punc=True, remove_url=True, remove_spaces=True, lower=True):
        """Clean the text
        
        Arguments:
            text {string} -- the text we want to clean
        
        Keyword Arguments:
            remove_emojis {bool} -- remove emojis from our text (default: {True})
            remove_numbers {bool} -- remove numbers from our text (default: {True})
            remove_punc {bool} -- remove punctuation from our text (default: {True})
            remove_url {bool} -- remove url's from our text (default: {True})
            remove_spaces {bool} -- remove extra spaces from our text (default: {True})
            lower {bool} -- make the text in lower case
        
        Returns:
            string -- the text after cleaning 
        """        

        if type(text) != str:
            return str(text)
        else:
            if lower:
                text = text.lower()
            if remove_spaces:
                nl_re = re.compile(r'(\n+)')
                text = re.sub(nl_re, ' ', text)
                t_re = re.compile(r'(\t+)')
                text = re.sub(t_re, ' ', text)
            if remove_url:
                url_re = re.compile("""((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?""")
                text = re.sub(url_re, " ", text)
            if remove_punc:
                text = text.translate(str.maketrans(' ', ' ', string.punctuation))
            if remove_numbers:
                numbers_re = re.compile(r'^\d+\s|\s\d+\s|\s\d+$')
                text = re.sub(numbers_re, ' ', text)
            if remove_emojis:
                text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
            return text

def create_tfds_dataset(texts, lables):
    """create tfds dataset

    Args:
        texts ([list]): [list of strings/pandas column]
        lables ([list]): [list of strings/pandas column]

    Returns:
        [tfds]: [tensorflow dataset object]
    """    
    ds = tf.data.Dataset.from_tensor_slices((texts, texts))
    return ds


class BERTGradientsScores():
    """plot gradient score per token in BERT prediction
    """    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_matrix = self.model.bert.embeddings.word_embeddings

    def get_grad(self, text):
        """calculate the gradients

        Args:
            text (string): the text to extract gradients for

        Returns:
            gradients, token_words, token_types: the gradients, token words and token types
        """        
        encoded_tokens =  self.tokenizer.encode_plus(text,\
                                                     add_special_tokens=True,\
                                                     return_token_type_ids=True,\
                                                     return_tensors="tf")
                                                     
        token_ids = list(encoded_tokens["input_ids"].numpy()[0])
        vocab_size = self.embedding_matrix.get_shape()[0]

        token_ids_tensor = tf.constant([token_ids], dtype='int32')
        token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(token_ids_tensor_one_hot)
        
            # multiply input model embedding matrix; this allows us do backprop wrt one hot input
            inputs_embeds = tf.matmul(token_ids_tensor_one_hot,self.embedding_matrix)  

            scores = self.model({"inputs_embeds": inputs_embeds,\
                                 "token_type_ids": encoded_tokens["token_type_ids"],\
                                 "attention_mask": encoded_tokens["attention_mask"] })
            gradient_non_normalized = tf.norm(
            tape.gradient([scores], token_ids_tensor_one_hot),axis=2)

            gradient_tensor = (
                gradient_non_normalized /
                tf.reduce_max(gradient_non_normalized)
            )
            gradients = gradient_tensor[0].numpy().tolist()
            
            token_words = self.tokenizer.convert_ids_to_tokens(token_ids) 
            token_types = list(encoded_tokens["token_type_ids"].numpy()[0])
        return gradients, token_words, token_types

    @staticmethod
    def plot_gradients(tokens, token_types, gradients, title):
        """plot the gradients score

        Args:
            tokens (list): token list
            token_types (list): token types
            gradients (list): gradients
            title (string): the title for the plot
        """        
        plt.figure(figsize=(21,3))
        xvals = [ x + str(i) for i,x in enumerate(tokens)]
        colors =  [ (0,0,1, c) for c,t in zip(gradients, token_types)]
        edgecolors = [ "black" if t==0 else (0,0,1, c)  for c,t in zip(gradients, token_types)]
        # colors =  [  ("r" if t==0 else "b")  for c,t in zip(gradients, token_types) ]
        plt.tick_params(axis='both', which='minor', labelsize=29)
        p = plt.bar(xvals, gradients, color=colors, linewidth=1, edgecolor=edgecolors)
        plt.title(title) 
        p=plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90) 

    def plot(self, text):
        """call the get grad and then plot gradients functions

        Args:
            text (string): the text to analyze
        """        
        gradients, token_words, token_types = self.get_grad(text)
        self.plot_gradients(token_words, token_types, gradients, text)

class TextAugmentation():
    def __init__(self, gensim_model='glove-twitter-25'):
        print(f"Downloading {gensim_model} model from gensim")
        self.gensim_model = api.load(gensim_model)
        self.transformers_nlp = pipeline('fill-mask')
        self.translator = Translator()
        print(f"Downloading wordnet from nltk")
        nltk.download("wordnet")

    def get_most_similar(self, word, topn=5):
        """get most similar words

        Args:
            word (string): word to compare in the model
            topn (int, optional): number of similar words to return. Defaults to 5.

        Returns:
            list: list of tuple (word, score)
        """        
        try:
            return self.gensim_model.most_similar(word, topn=topn)
        except:
            print(f"word {word} not in vocabulary")
            return []

        """return mask language model output (for example 'This is <mask> cool' -> 'This is pretty cool')

        Args:
            text ([string]): a string to feed the MLM
        """
    def get_MLM(self, text):
        """return mask language model output (for example 'This is <mask> cool' -> 'This is pretty cool')

        Args:
            text ([string]): a string to feed the MLM

        Returns:
            list: list of dictionaries {'score': float, 'sequence': string, 'token': int, 'token_str': string}
        """        
        return self.transformers_nlp(text)

    @staticmethod
    def get_synonym(text):
        """return the synonym of a word

        Args:
            text (string): a word

        Returns:
            list: list of words that are synonym to the word
        """        
        synonyms = []
        for syn in wordnet.synsets(text):
            for lm in syn.lemmas():
                    if lm.name().lower() != text:
                        synonyms.append(lm.name().lower())
        return synonyms
    
    def get_translation(self, text, dest_lang, circle=False, orig_lang=None):
        """translate text using google API, for all the language list visit: https://py-googletrans.readthedocs.io/en/latest/

        Args:
            text (list): list of strings
            dest_lang (satring): string that describe the desired language
            circle (bool, optional): whether to to circle translation (en -> fr -> en). Defaults to False.
            orig_lang ([type], optional): the original language of the text (circle=True). Defaults to None.       

        Raises:
            ValueError: error if the input is not a list

        Returns:
            list: list of string of the translation
        """    
        if isinstance(text, list):
            if not circle:
                    translation = self.translator.translate(text, dest=dest_lang)
                    translation_list = []
                    for item in  translation:
                        translation_list.append(item.text)
                    return translation_list
            else:
                    translation = self.translator.translate([x.text for x in self.translator.translate(text, dest=dest_lang)], dest=orig_lang)
                    translation_list = []
                    for item in  translation:
                        translation_list.append(item.text)
                    return translation_list
        else:
            raise ValueError(f"{type(text)} provided, supporting type {list}")