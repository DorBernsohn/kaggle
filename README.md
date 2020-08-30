# kaggle
kaggle projects
| Name | url | Repo name |
| --- | --- | --- |
| `imdb-dataset-of-50k-movie-reviews` | https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews | Deep_IMDB |
| `Jigsaw Multilingual Toxic Comment Classification` | https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification | Multilingual_Toxic_Comment_Classification |
| `iMaterialist (Fashion) 2020 at FGVC7` | https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7 | imaterialist_fashion_2020 |
| `Tweet Sentiment Extraction` | https://www.kaggle.com/c/tweet-sentiment-extraction | Tweet_Sentiment_Extraction |
| `Reddit selfposts classification` | https://www.kaggle.com/mswarbrickjones/reddit-selfposts | Reddit_selfposts_classification |
| `Contradictory, My Dear Watson` | https://www.kaggle.com/c/contradictory-my-dear-watson | Contradictory_classification |
| `google-quest-challenge` | https://www.kaggle.com/c/google-quest-challenge | google_quest_challenge |


<details>
<summary>utils description</summary>

+ setup.py
    + set_TPU: config TPU for tensorflow
    + methdispatch: Adjustment of @singledispatchmethod usage to a python version lower than 3.8.
+ text.py
    + clean_text: Clean the text
    + create_tfds_dataset: create tfds dataset
    + textAugmentation: perform text augmentation
        + get_most_similar: get most similar words
        + get_MLM: return mask language model output (for example 'This is <mask> cool' -> 'This is pretty cool')
        + get_synonym: return the synonym of a word
        + get_translation: translate text using google API
+ text_models.py
    + BertInputs: create a BERT inputs
    + XLNetInputs: create a XLNet inputs and build a model
+ vision.py
    + refine_masks: refine the mask to avoid overlapping
    + resize_image: resize the image by the config settigs