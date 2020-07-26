 * @author  Dor Bernsohn

import re
import emoji
import string
from tqdm import tqdm

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
