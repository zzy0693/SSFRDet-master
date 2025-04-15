# SSFRDet

## Introduction

Semantic and Spatial Feature Reinforcement for Object Detection
![](C:\Users\admin\Desktop\model.jpg)

## Prerequisites

Win10;
Python 3.8+;
PaddleDetection=2.8.0;
CUDA 11.2+ (If you build Paddle from source, CUDA 11.1 is also compatible);
Paddle=2.4.2;

## Model Zoo

| Backbone | Model | Images/GPU | Inf time (fps) | Box AP |   Config    | Download |
|:------:|:--------:|:----------:|:--------------:|:------:|:-----------:|:--------:|
| R-50 | SSFRDet  |     1      |     14.09      |  44.1  | config file | [model](https://drive.google.com/file/d/1ZWp9BNRXFazBjsXgTxEdwyo61pWii3xQ/view?usp=drive_link) |

## Run command

The run commands still follow the PaddleDetection library command format. For example, the training model configuration command: python train.py -c configs/tood/tood_r50_fpn_1x_coco.yml --eval.

**Notes:**

- SSFRDet is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.
- SSFRDet uses GPU-V100 to train 12 epochs or 24 epochs.

GPU single-card training

```
Acknowledgement
The implementation of SSFRDet is based on PaddleDetection.

```
