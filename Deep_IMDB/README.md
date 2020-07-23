# Deep_IMDB
Experiment on few DL models (pre-trained and regular) on the IMDB dataset

| Model | Loss | Accuracy | Test/Train ratio | Trainable Params |
| --- | --- | --- | --- | --- |
| `BiLSTMs + Self Attention` | 0.2440 | 0.9049 | 0.2 | 6,657,050 | 
| `gnews-swivel-20dim-with-oov/1` | 7.5760 | 0.5031 | 0.2 | 389,821 |
| `gnews-swivel-20dim/1` | 0.4763 | 0.7937 | 0.2 | 400,461 |
| `nnlm-en-dim50/1` | 0.3700 | 0.8736 | 0.2 | 48,193,201 |
| `nnlm-en-dim128/1` | 0.3724 | 0.8531 | 0.2 | 124,659,329 |
| `universal-sentence-encoder/4` | 0.3280 | 0.8922 |0.2 | 257,060,993 |
| `bert-base-uncased` | 0.1725 | 0.9346 | 0.5 | 109,483,778 |
| `xlnet-base-cased` | 0.1590  | 0.0.9439 | 0.5 | 117,310,466 |

