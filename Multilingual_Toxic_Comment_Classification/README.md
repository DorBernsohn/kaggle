# Jigsaw Multilingual Toxic Comment Classification

Experiment on few DL models (pre-trained and regular) on the Jigsaw Multilingual Toxic Comment Classification dataset

| Model | Validation Loss | Validation Accuracy | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- | --- |
| `bert-base-multilingual-uncased` | 0.6315 | 0.8497 | 0.0357 | 167,357,954 |
| `bert-base-multilingual-uncased - 2nd phase` | 0.2445 | 0.8920 | 0.33 | 167,357,954 |

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
    + 2nd phase training
</details>