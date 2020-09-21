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


class TextAugmentation():
    def __init__(self, gensim_model='glove-twitter-25'):
        print(f"Downloading {gensim_model} model from gensim")
        self.gensim_model = api.load(gensim_model)
        self.transformers_nlp = pipeline('fill-mask', device=-1)
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