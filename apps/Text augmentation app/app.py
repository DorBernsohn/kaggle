import os
import json
import string
import random
import logging
import warnings
import requests
import numpy as np
from setup import TextAugmentation
from flask_bootstrap import Bootstrap
from flask import Flask, request, redirect, url_for, render_template

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
Bootstrap(app)

"""
Constants
"""

"""
Utility functions
"""

"""
Routes
"""
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form["text"]
        method = request.form.get("methodDropdown")
        if method == "mostSimilarWords":
            gms = [x[0] for x in aug.get_most_similar(text)]
            result = {
                        'word': text,
                        'pred': ', '.join([x for x in gms]),
                        "method": "Most similar words:"
                    }
        elif method == "synonyms":
            gs = aug.get_synonym(text)
            result = {
                        'word': text,
                        'pred': ', '.join([x for x in gs]),
                        "method": "Synonyms words:"
                    }
        elif method == "translation":
            orig_lang = request.form.get("destLangDropdown")
            dest_lang = request.form.get("origLangDropdown")
            gt = aug.get_translation([text],dest_lang=dest_lang, orig_lang=orig_lang, circle=True)
            result = {
                        'word': text,
                        'pred': ', '.join(x for x in gt),
                        "method": "Translation:"
                    }
        elif method == "MLM":
            gmlm = aug.get_MLM(text)
            elements = ', '.join([x['token_str'][1:] if x['token_str'].startswith("Ġ") else x['token_str'] for x in gmlm])
            result = {
                        'word': text,
                        'pred': elements,
                        "method": "Mask word prediction:"
                    }
        else:
            result = {
                        'word': text,
                        'pred': '',
                        "method": "Choose a method."
                    }
        
        return render_template('show.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    logging.debug("Init TextAugmentation")
    aug = TextAugmentation()
    app.run(debug=True)