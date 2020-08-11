 # @author  DBernsohn

import re
import nltk
import emoji
import string
import warnings
import tensorflow as tf
import gensim.downloader as api
from nltk.corpus import wordnet
from transformers import pipeline
from googletrans import Translator

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')  
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='transformers')

def clean_text(text, remove_emojis=True, remove_numbers=True, remove_punc=True, remove_url=True, remove_spaces=True):
        """Clean the text
        
        Arguments:
            text {string} -- the text we want to clean
        
        Keyword Arguments:
            remove_emojis {bool} -- remove emojis from our text (default: {True})
            remove_numbers {bool} -- remove numbers from our text (default: {True})
            remove_punc {bool} -- remove punctuation from our text (default: {True})
            remove_url {bool} -- remove url's from our text (default: {True})
            remove_spaces {bool} -- remove extra spaces from our text (default: {True})
        
        Returns:
            string -- the text after cleaning 
        """        

        if type(text) != str:
            return str(text)
        else:
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
    """create tensorflow dataset

    Args:
        texts ([list]): [list of strings/pandas column]
        lables ([list]): [list of strings/pandas column]

    Returns:
        [tfds]: [tensorflow dataset object]
    """    
    ds = tf.data.Dataset.from_tensor_slices((texts, texts))
    return ds

class textAugmentation():
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