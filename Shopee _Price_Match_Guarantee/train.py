import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19

from models import models
from utils import Preprocess

def load_data(data_dir: str) -> pd.DataFrame:
    """load_data loads the data into a pandas dataframe

    Args:
        data_dir (str): the path to the csv file

    Returns:
        pd.DataFrame: pandas dataframe
    """    
    train_img_dir = data_dir + os.sep + 'train_images'
    data = pd.read_csv(f'{data_dir}{os.sep}filtered_train.csv')
    data = data.sample(frac=1)
    label_group_mapping = {v: i for i, v in enumerate(set(data.label_group.values))}

    data['path'] = train_img_dir + os.sep + data.image
    data['new_label'] = data.label_group.apply(lambda x: label_group_mapping.get(x))
    return data
    
if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)
    data_dir = '/shopee'
    data = load_data(data_dir)
    print("Data has been Loaded")
    vision_model = VGG19(weights="imagenet", include_top=False, pooling='avg')
    preprocess = Preprocess(data, '/shopee/train_images/', vision_model=vision_model)
    preprocess.preprocess()
    print("Data has been Preprocess")

    for model_name in models.keys():
        print(f"Training {model_name}")
        os.system(f"mkdir -p {model_name}")
        trainer = models[model_name](num_labels=data.new_label.nunique())
        history = trainer.train(preprocess.data, vision_model=vision_model)
        print("Model has been Trained")
        trainer.save(f"{model_name}/")
        with open(f'{model_name}/trainHistoryDict_{model_name}', 'wb') as f:
                pickle.dump(history.history, f)
        del trainer
        del history