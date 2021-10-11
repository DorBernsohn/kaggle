# Shopee - Price Match Guarantee
[Competition url](https://www.kaggle.com/c/shopee-product-matching)

# Data
+ To get the data using kaggle dataset: `kaggle competitions download -c shopee-product-matching`

#### The images, with titles, looks like:

![image1](images/image1.jpg?raw=true)
![image2](images/image2.jpg?raw=true)
![image3](images/image3.jpg?raw=true)
![image4](images/image4.jpg?raw=true)

# Data preprocessing
+ The main problem here is to have two similar items that labeled with two different `label_group`. To adress this i used BKTree to check for distances among the phash.

# Modeling
+ I trained a classifier, for each image and title, to predict which `label_group` the product is.
  + I compared 2 different techniques:
    + Softmax
    + ArcFace
  + The model architecture is as the figure describes:


![Architecture](images/architecture.png?raw=true)

# Package information
+ tensorflow: 2.4.1
+ transformers: 4.11.2

# Train
+ `python train.py`
