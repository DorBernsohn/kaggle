# reddit self-post classification

Experiment on few DL models (pre-trained and regular) on the The reddit self-post classification task dataset

| Model | Validation Loss | Validation Accuracy | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- | --- |
| `bert-base-multilingual-uncased` | 0.4399 | 0.8761 | 0.25 | 167,391,790 |
| `xlm-roberta-base` |  |  |  |  |


<details>
<summary>Table of content</summary>

+ Imports and TPU setting
+ Load the data
+ Preprocess
+ Modelling
    + Build model inputs
    + Build model
    + Training
    + Load model
</details>