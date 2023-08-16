# Zero-shot Nuclei Detection via Visual-Language Pre-trained Models

<img src="VLPMNuD.png" width="800">

Official implementation of Zero-shot Nuclei Detection via Visual-Language Pre-trained Models.
The original paper link is here:
[arXiv link](to be update), [MICCAI link](to be update).
The proposed method has two process steps:
1.Generating raw GLIP prediction result.
2.Self-training via YOLOX .
This repository provides source code for the first step.Code for second step is [here](https://github.com/wuyongjianCODE/VLPMNuD_part2).
## Installation
1.Create a Python environment based on ```requirements.txt``` file.

2.Build up GLIP.
Our project is developed based on [GLIP](https://github.com/microsoft/GLIP).

```bash 
cd sick/GLIP
python setup.py develop
```
3.[Optional] We recommend to install transformer libiary from source code , which enable us to change some function inside the transformer backbone.If you have installed transformer via pip and only want to reproduce the experiment result in our paper, you can also skip this step.This option is intended for further development.
You firstly need to uninstall transformer via pip or conda.
Then:
```bash 
cd github_src/transformers
python setup.py develop
```
## Dataset Introduction

For the convenience of downloading and using, we directly placed the dataset inside the project.It is located in DATASET/coco.For a dataset of tens of MB, it is far more convenient to use ```git clone``` directly than to download it from a network disk.
It was originally a MoNuSAC dataset and has been customized into COCO dataset format by us. This path will be directly called in the code, so it is not recommended to modify its path.
Making it easy for others to reproduce initial results is the original intention of our team, sincerely.

## Test : Generate the raw GLIP prediction result
Before testing, please download pretrain weights of [GLIP large model](https://huggingface.co/GLIPModel/GLIP/blob/main/glip_large_model.pth).
Then:
```bash 
python tools/test_grounding_net_wuap.py --config-file configs/pretrain/glip_Swin_L.yaml --weight glip_large_model.pth 
--prompt 'rectangular black nuclei. circle black nuclei. spherical black nuclei. rectangular dark purple nuclei. circle dark purple nuclei. spherical dark purple nuclei.' TEST.IMS_PER_BATCH 1 MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR OUTPUT
```
Caution: ```--prompt``` param here gives an example prompt obtained from [blip](to be update), you can also input other prompts.
A json file recording all bbox will be generated in ```./jsonfiles```.Normally, the mAP score of raw glip should be in range [0.16, 0.2].Then you can use the predict result (i.e. the json file) to run self-training algorithm, and you will get near SOTA score (range [0.4, 0.45]).We provide our self-training source code [here](https://github.com/wuyongjianCODE/VLPMNuD_part2).


## Citing VLPM
If you use VLPM in your work or wish to refer to the results published in this repo, please cite our paper:
```BibTeX

```



