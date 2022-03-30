# TricubeNet - Official Pytorch Implementation (WACV 2022)

**TricubeNet: 2D Kernel-Based Object Representation for Weakly-Occluded Oriented Object Detection** <br />
Beomyoung Kim<sup>1</sup>, Janghyeon Lee<sup>2</sup>, Sihaeng Lee<sup>2</sup>, Doyeon Kim<sup>3</sup>, Junmo Kim<sup>3</sup><br>

<sup>1</sup> <sub>NAVER CLOVA</sub><br />
<sup>2</sup> <sub>LG AI Research</sub><br />
<sup>3</sup> <sub>KAIST</sub><br />

WACV 2022 <br />

[![Paper](https://img.shields.io/badge/arXiv-2104.11435-brightgreen)](https://arxiv.org/abs/2104.11435)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tricubenet-2d-kernel-based-object/one-stage-anchor-free-oriented-object-1)](https://paperswithcode.com/sota/one-stage-anchor-free-oriented-object-1?p=tricubenet-2d-kernel-based-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tricubenet-2d-kernel-based-object/one-stage-anchor-free-oriented-object-4)](https://paperswithcode.com/sota/one-stage-anchor-free-oriented-object-4?p=tricubenet-2d-kernel-based-object)

<img src = "https://github.com/qjadud1994/TricubeNet/blob/main/figures/overview.png" width="100%" height="100%">


## How to use?

#### Data Preparation

* [DOTA official page](https://captain-whu.github.io/DOTA/index.html)

#### For training
	bash run_train.sh

Please check the discription of training hyperparameters (we recommend to use default hyperparameters)

	python3 train.py --help


#### For testing

	cd evaluation
	bash run_eval.sh

Please check the discription of testing hyperparameters (we recommend to use default hyperparameters)

	python3 eval_DOTA.py --help


## Qualitative Results

#### DOTA
<img src = "https://github.com/qjadud1994/TricubeNet/blob/main/figures/DOTA.png" width="70%" height="70%">

#### MSRA-TD500, ICDAR 2015
<img src = "https://github.com/qjadud1994/TricubeNet/blob/main/figures/Text-Detection.png" width="70%" height="70%">

#### SKU110K-R
<img src = "https://github.com/qjadud1994/TricubeNet/blob/main/figures/SKU110K-R.png" width="70%" height="70%">



## Citation
We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
~~~
@inproceedings{kim2022tricubenet,
  title={TricubeNet: 2D Kernel-Based Object Representation for Weakly-Occluded Oriented Object Detection},
  author={Kim, Beomyoung and Lee, Janghyeon and Lee, Sihaeng and Kim, Doyeon and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={167--176},
  year={2022}
}
~~~
