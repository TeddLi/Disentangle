# Disentangle
This repository contains the source code and data for the paper:
DialBERT: A Hierarchical Pre-Trained Model for Conversation Disentanglement
https://arxiv.org/abs/2004.03760
## Dependencies

Python 3.6
Tensorflow 1.14.0


## Results


## Get adaptation and Pre-train checkpoints

This is Adaptation model

https://drive.google.com/drive/folders/1n6ymYqM1xv08mWW0I-kHIBylK4VR6W-F?usp=sharing

This is BERT-base Pre-trained checkpoints

https://drive.google.com/drive/folders/13Q0OtsRKSAEilt4U-ycQBa4CWFsGHbpv?usp=sharingcd ..




## Train a new model
```
cd Disentangle/scripts/bash_file/
bash train_exloss.sh
```

## Get the results of all evaluation matrics

```
cd Disentangle/scripts/ckpt/your_ckpt
base valid.sh
```


## Cite
```
@article{DBLP:journals/corr/abs-2004-03760,
  author    = {Tianda Li and
               Jia{-}Chen Gu and
               Xiaodan Zhu and
               Quan Liu and
               Zhen{-}Hua Ling and
               Zhiming Su and
               Si Wei},
  title     = {DialBERT: {A} Hierarchical Pre-Trained Model for Conversation Disentanglement},
  journal   = {CoRR},
  volume    = {abs/2004.03760},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.03760},
  archivePrefix = {arXiv},
  eprint    = {2004.03760},
  timestamp = {Tue, 14 Apr 2020 16:40:34 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-03760.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
