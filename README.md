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

## Synthetic Dataset Generation Code

- [ ] TODO

## To-Do List

Excluding what was discussed in the Future Work section of the paper, here are some outstanding tasks yet to be completed. Commits to this repository will be made once fixed.

- [ ] MMDetection and standalone Keras implementations of **a YOLOv3 TSDA model has been partially implemented**.

- [ ] Resolve/merge training script related differences between non-TSDA and TSDA versions of EfficientDet where appropriate.

- [ ] Fix TSDA model with `num_damage_sectors=1` ($m=1$), currently only partially implemented and still has errors.

### DICTA Submission TODO

***Delete this soon***

- [ ] Directory for dataset generation code (don't forget damage-related eval code)? If so, leave comment in README here that a subsequent publication will cover further features of code

- [ ] Provide permalinks to the templates and backgrounds used for generating the synthetic dataset (cite Geograph contributors)

- [ ] Provide permalinks to the exact datasets used in the paper's experiments (5 synth levels and 12000_synth_test)

- [ ] Provide permalink to the Singularity container and/or container definition (may be able to use Pawsey/Singularity cloud thing)

- [x] Directory for sign detection version of EfficientDet

- [x] Directory for TSDA version of EfficientDet

- [x] Show experiment training details and results in README
