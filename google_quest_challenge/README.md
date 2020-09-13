# google-quest-challenge

Experiment on few DL models (CLS based and siamse) on the google-quest-challenge

| Model | Validation Loss | Validation Accuracy | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- | --- |
| `jplu/tf-xlm-roberta-large - CLS based` |  |  |  | 560,430,622 |
| `bert-base-uncased` |  | |  | 109,891,358 |
| `bert-base-uncased - siamse`|  |  |  | 110,284,574 |


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