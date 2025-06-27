# From Dense to Sparse: Event Response for Enhanced Residential Load Forecasting (TIM 2025)

> **IMPORTANT**  
> Thank you for your interest in our project! If you find this work useful, please give it a ⭐ Star on GitHub to show your support.
## Table of Contents
1. [Introduction](#introduction)
2. [Quickstart](#quickstart)
3. [Execution Example](#execution-example)
4. [Data and Models](#data-and-models)
5. [Citation](#citation)
6. [Acknowledgement](#acknowledgement)
7. [Contact](#contact)
---
## Introduction
This repository contains the implementation for our IEEE TIM 2025 paper "From Dense to Sparse: Event Response for Enhanced Residential Load Forecasting". We propose a novel framework that improves residential load forecasting accuracy by leveraging sparse event responses to capture appliance usage patterns.
---
## Quickstart
To set up the environment and dependencies like build.sh.
### Execution Example
Train MSP model (model 2),Enhance RLF with MSP (model 3).
``` bash
python knowledge4tsf/main.py \
    --model 3 \
    --device cuda:0 \
    --tsf_model patchTsMixer \
    --pred_len 1 \
    --status_file status_model1_predL_1_horizon_1_topk_22_umass3.pth \
    --data umass3 \
    --data_dim 22 \
    --topk 22 \
    > /tmp/umass3_1_patchTsMixer.log

```
## Data and Models
[Google Drive Download](https://drive.google.com/drive/folders/1Jyxv_fihu-kZ0yQiO0SGPCEeIBpGcrte)


## Citation
```bibtex
@article{cao2025erkg,
  title={From Dense to Sparse: Event Response for Enhanced Residential Load Forecasting},
  author={Cao, Xin and Tao, Qinghua and Zhou, Yingjie and Zhang, Lu and Zhang, Le and Song, Dongjin and Oliver Wu, Dapeng and Zhu, Ce},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  volume={74},
  pages={1-12},
  year={2025},
  doi={10.1109/TIM.2025.3544349}
}

```
## Acknowledgement
We acknowledge these open-source projects:

- [RevIN](https://github.com/ts-kim/RevIN) - Reversible Instance Normalization
- [ETSformer](https://github.com/salesforce/ETSformer) - Time Series Forecasting Transformer
- [AttrFaceNet](https://github.com/FVL2020/AttrFaceNet)
## Contack
For questions and collaborations:

Xin Cao: caoxin9629@gmail.com
