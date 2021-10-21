# visualization utils

import cv2
import pandas as pd
import matplotlib.pyplot as plt

def image_vis(image_path, title=None, ax=None):
    """
    Function for visualization.
    Takes path to image as input.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ax:
        ax.imshow(img)
        ax.set_title(title)
    else:
        plt.imshow(img)
        plt.axis('off')

# phash utils

import imagehash
from tqdm import tqdm

def match_matrix(phash_array):
    """
    A function that checks for matches by phash value.
    Takes phash values as input.
    Output - phash diff matrix (pandas data frame)
    """
    phashs = phash_array.apply(lambda x: imagehash.hex_to_hash(x))
    phash_matrix = pd.DataFrame()
    for i in tqdm(phash_array):
        phash_matrix = pd.concat([phash_matrix, phashs - imagehash.hex_to_hash(i)], 
                                 axis = 1)
    phash_matrix.columns = range(len(phash_array))
    return phash_matrix

# triplet loss

import random
import tensorflow as tf

def generate_triplets(df):
    random.seed(42)
    group_df = dict(list(df.groupby('label_group')))

    image_group_df = dict(list(df.groupby('image')))
    title_group_df = dict(list(df.groupby('title')))

    def aux(row):
        anchor = row.posting_id
        
        ids = group_df[row.label_group].posting_id.tolist()
        ids.remove(row.posting_id)
        for posting_id in image_group_df[row.image].posting_id.tolist():
            if posting_id in ids:
                ids.remove(posting_id)
        for posting_id in title_group_df[row.title].posting_id.tolist():
            if posting_id in ids:
                ids.remove(posting_id)
        if len(ids) == 0:
            return None, None, None
        positive = random.choice(ids)
        
        groups = list(group_df.keys())
        groups.remove(row.label_group)
        neg_group = random.choice(groups)
        negative = random.choice(group_df[neg_group].posting_id.tolist())

        return anchor, positive, negative
    
    return aux

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    target_shape = (200, 200)
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

def build_dataset(paths):
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(preprocess_image, num_parallel_calls=AUTO)
    dset = dset.cache()

    # Apply batching
    dset = dset.batch(32).prefetch(AUTO)
    return dset

def visualize(train_triplets_imgs, size):
    """Visualize a few triplets from the supplied batches."""
    sample_df = train_triplets_imgs.sample(size)
    anchor = sample_df['anchor'].values.tolist()
    positive = sample_df['positive'].values.tolist()
    negative = sample_df['negative'].values.tolist()

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(size, 3)
    for i in range(size):
        show(axs[i, 0], plt.imread(anchor[i]))
        show(axs[i, 1], plt.imread(positive[i]))
        show(axs[i, 2], plt.imread(negative[i]))

class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]