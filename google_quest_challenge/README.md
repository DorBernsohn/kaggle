# google-quest-challenge

Experiment on few DL models (CLS based and siamse) on the google-quest-challenge

| Model | Validation Loss  | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- |
| `jplu/tf-xlm-roberta-large - CLS based` | 0.3804 | 0.2 | 560,430,622 |
| `bert-base-uncased - CLS based` | 0.4254 | 0.2 | 109,891,358 |
| `bert-base-uncased - siamse`| 0.3773 | 0.2 | 110,284,574 |


<details>
<summary>Table of content</summary>

+ Imports and TPU setting
+ Load the data
+ Preprocess
+ Modelling
    + Model by CLS token - xlm-roberta-large
        + Build model inputs
        + Build model
        + Training
    + Model by CLS token - BERT
        + Build model inputs
        + Build model
        + Training
    + Model by BERT siamese network
        + Build model inputs
        + Build model
        + Training
</details>