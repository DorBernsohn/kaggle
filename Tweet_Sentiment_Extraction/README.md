# Tweet Sentiment Extraction

Experiment on few DL models (pre-trained and regular) on the Tweet Sentiment Extraction task

| Model | Loss | Jaccard | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- | --- |
| `roberta-base` | 1.6273  | 0.6972 | 0.2 | 124,647,170 |

Jaccard = ` def jaccard(str1, str2): 
                a = set(str1.lower().split()) 
                b = set(str2.lower().split())
                c = a.intersection(b)
                return float(len(c)) / (len(a) + len(b) - len(c))
          `
to be complete in few weeks
<details>
<summary>Table of content</summary>

+ Imports and TPU setting
+ Load the data
+ Text augmentation
    + TODO
+ Preprocess
+ Modelling
    + Build model
+ Training
</details>
