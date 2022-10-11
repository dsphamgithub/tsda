# End-to-End Traffic Sign Damage Assessment

By *Kristian Radoš, Jack Downes, Duc-Son Pham,* and *Aneesh Krishna*

Model training information and frozen copy of codebase from a [paper submitted for publication]() in *The International Conference on Digital Image Computing: Techniques and Applications (DICTA) 2022 Sydney*.

## Abstract
> Traffic sign damage monitoring is a practical issue facing large operations all over the world. Despite the scale of traffic sign damage and its consequent impact on public safety, damage audits are performed manually. By automating components of damage assessment we can greatly improve the effectiveness and efficiency of the process and in doing so alleviate its negative impact on traffic safety. In this paper, traffic sign damage assessment is explored as a computer vision problem approached with deep learning. We specifically focus on occlusion-type damages that hinder sign legibility. This paper makes several contributions. Firstly, it provides a comprehensive survey of related work on this problem. Secondly, it provides an extension to the generation of synthetic images for such a study. Most importantly, it proposes an extension of the EfficientDet object detection framework to address the challenge. It is shown that synthetic images can be successfully used to train an object detector variant to assess the level of damage, as measured between 0.0 and 1.0, in traffic signs. The extended framework achieves a damage assessment root mean squared error (RMSE) of 0.087 on a synthetic test set while maintaining a mean average precision (mAP) of 86.3% on the typical sign detection task.

### BibTex
```
TODO

@inproceedings{___,
  title={End-to-End Traffic Sign Damage Assessment},
  author={Radoš, Kristian and Downes, Jack and Pham, Duc-Son and Krishna, Aneesh},
  booktitle={___},
  pages={___},
  year={2022}
}
```

## EfficientDet Experiments
The code for the EfficientDet models is divided into two directories, `EfficientDet_No_TSDA` as used for the standard sign detection experiments and `EfficientDet_TSDA`, the model that has been modified to perform Traffic Sign Damage Assessment (TSDA).

See the README in each model directory for experiment training configurations and evaluation results.

All experiments were trained and evaluated on the Pawsey Supercomputer using Singularity containers for reproducibility. The `.def` file is below and the [full container is here]().
```
Bootstrap: docker
From: nvcr.io/nvidia/tensorflow:21.05-tf2-py3
Stage: build
%post
    pip install 'lxml>=4.6.1'
    pip install 'pandas'
    pip install 'absl-py>=0.10.0'
    pip install 'matplotlib>=3.0.3'
    pip install 'numpy>=1.19.4'
    pip install 'Pillow>=6.0.0'
    pip install 'PyYAML>=5.1'
    pip install 'six>=1.15.0'
    pip install 'tensorflow==2.4.0'
    pip install 'tensorflow-addons>=0.12'
    pip install 'tensorflow-hub>=0.11'
    pip install 'neural-structured-learning>=1.3.1'
    pip install 'tensorflow-model-optimization>=0.5'
    pip install 'Cython>=0.29.13'
    pip install 'pycocotools==2.0.3'
    pip install 'opencv-python'
    pip install 'scikit-image'
    pip install 'imutils'
    pip install 'plotly'
    pip install 'tqdm'
    pip install 'wandb'
```

## Synthetic Dataset Generation
[[download]](https://drive.google.com/file/d/19FuxvFtov2gFGeZEaOndEO3Qb9YkExx8/view?usp=sharing) | The complete collection of datasets used for the above EfficientDet experiments.

The `test` set directories are equivalent to the original GTDSB test set. The synthetic images used for the `_extended` datasets were taken from the same pool of synthetic images. The `12000_synth_test` synthetic images were generated using the same set of templates and backgrounds  as the `_extended` datasets, but they share no images, i.e. they are independent.

### Traffic Sign Templates
[[download]](https://drive.google.com/file/d/1kMAPRSOs9RqAtQu6-fUEn1fqkazIC3Kt/view?usp=sharing) | GTSDB classes matched using German [Wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany) and [Wikimedia Commons](https://commons.wikimedia.org/wiki/Historic_road_signs_in_Germany#1992%E2%80%932013) images. Covers 43/43 classes.

### Backgrounds
[[download]](https://drive.google.com/file/d/1navoOiHRhYhrIGgogp1TMoEg3QczvN85/view?usp=sharing) | 1191 images taken from various sources with no visible traffic sign faces.

A set of 1,191 traffic scene backgrounds gathered from 4 sources were used to generate the synthetic dataset. All were filtered so at to contain no unlabelled real traffic signs. The different sources are described under the below headings.

#### Google Street View
925 images from countries surrounding Germany pulled from the Google Street View API using Hugo van Kemenade's [random-street-view](https://github.com/hugovk/random-street-view). The breakdown by country is as follows:
| Code        | Country           | Images  |
| ------------- |-------------| -----:|
AUT | Austria | 100
BEL | Belgium | 100
CHE | Switzerland | 100
CZE | Czechia | 100
DEU | Germany | 25
DNK | Denmark | 100
FRA | France | 100
GBR | United Kingdom | 50
LUX | Luxembourg | 50
NLD | Netherlands | 100
POL | Poland | 100

#### Cityscapes
191 images from Germany were taken from the [Cityscapes dataset](https://www.cityscapes-dataset.com). They were chosen by automatically filtering out all images containing traffic signs using the ground truth labels provided with the dataset. The code used to do so can be found in `cityscapes_backgrounds.py`.

#### Geograph
50 images from the UK were manually picked out from [www.geograph.org.uk](https://www.geograph.org.uk). The webpage for each image can be found by searching on the website using the ID in its filename. 48 images were photographed by David Howard and 2 were photographed by Peter Wood. All credit goes to them.
> © Copyright [David Howard](https://www.geograph.org.uk/profile/6358) and licensed for reuse under [creativecommons.org/licenses/by-sa/2.0](https://creativecommons.org/licenses/by-sa/2.0/)

> © Copyright [Peter Wood](https://www.geograph.org.uk/profile/72434) and licensed for reuse under [creativecommons.org/licenses/by-sa/2.0](https://creativecommons.org/licenses/by-sa/2.0/)

#### Google Images
25 images, primarily from the UK and Germany, were found using Google Images. Google reverse image search can be used to find the original sources.

## To-Do List

Excluding what was discussed in the Future Work section of the paper, here are some outstanding tasks yet to be completed. Commits to this repository will be made once addressed.

- [ ] MMDetection and standalone Keras implementations of **a YOLOv3 TSDA model has been partially implemented**.

- [ ] Resolve/merge training script related differences between non-TSDA and TSDA versions of EfficientDet where appropriate.

- [ ] Fix TSDA model with `num_damage_sectors=1` ($m=1$), currently only partially implemented and still has errors.

### DICTA Submission TODO

***Delete this section when done***

- [x] Directory for dataset generation code (don't forget damage-related eval code)? If so, leave comment in README here that a subsequent publication will cover further features of code

- [x] Provide permalinks to the templates and backgrounds used for generating the synthetic dataset (cite Geograph contributors)

- [x] Provide permalinks to the exact datasets used in the paper's experiments (5 synth levels and 12000_synth_test)

- [ ] Provide permalink to the Singularity container and/or container definition (may be able to use Pawsey/Singularity cloud thing)

- [x] Directory for sign detection version of EfficientDet

- [x] Directory for TSDA version of EfficientDet

- [x] Show experiment training details and results in README
