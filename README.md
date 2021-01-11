# Single Shot Temporal Action Detection

A pytorch-version implementation codes of paper: "Single Shot Temporal Action Detection", which is accepted in ACM MM 2017. [paper](https://arxiv.org/abs/1710.06236)

This repository is an improved version for the anchor-based part of [A2Net](https://github.com/VividLe/A2Net).

## Result

The detection results on THUMOS14 dataset:

| mAP@ | 0.3  | 0.4  | 0.5  | 0.6  | 0.7  |
| :--: | :--: | :--: | :--: | :--: | :--: |
|      |      |      |      |      |      |

## Prerequisites

This repository is implemented with Pytorch 1.1.0 + Python3.

## Download Datasets

The Two stream I3D feature could be downloaded from [A2Net](https://github.com/VividLe/A2Net).

## Training and Testing of SSAD

1. To train the SSAD:

```
cd tools
python main.py
```

The parameters could be modified in 

```
experiments\thumos\SSAD_train.yaml
```

2. To Test the SSAD:

```
cd tools
python eval.py --checkpoint $cpt_path
```

3. Evaluating the detection performance:

Open Matlab in `lib\Evaluation\THUMOS14_evalkit_20150930` path, and put the testing result file in the path, and execute the file:

```
multi_iou_eval
```

