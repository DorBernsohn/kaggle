 # @author  DBernsohn

import re
import emoji
import string
from googletrans import Translator

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

def get_translation(text, dest_lang):
    """translate text using google API

    Args:
        text (string/list): string or list of strings
        dest_lang (satring): string that describe the desired language

    Raises:
        ValueError: error if the input is not string or list

    Returns:
        string\list: string or list of string of the translation
    """    
    translator = Translator()
    if isinstance(text, list):
        translation = translator.translate(text, dest=dest_lang)
        translation_list = []
        for item in  translation:
            translation_list.append(item.text)
        return translation_list
    elif isinstance(text, str):
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    else:
        raise ValueError(f"{type(text)} provided, supporting string or list")