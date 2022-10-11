# Synthetic Dataset Generation

We have modified [code](https://github.com/alexandrosstergiou/Traffic-Sign-Recognition-basd-on-Synthesised-Training-Data) by Alexandros Stergiou ([2018 paper](https://www.mdpi.com/2504-2289/2/3/19)) to create synthetic traffic signs which are placed over a background image to create a holistic data that can be used to train a detection model directly rather than training for simple classification. Furthermore, we have implemented various kinds of synthetic damage that are applied to the signs. This damage is quantified by using the percentage of the sign that has been changed or obscured. The purpose of this is to train a detection model to not only detect each class of sign, but to also detect the class and assess the severity of damage that is present on the sign, if any.
 
 ![damaged_examples](https://github.com/BunningsWarehouseOfficial/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data/blob/main/Figures/Damaged_examples.png "Templates")
 
The above shows some close-up examples of damage applied to signs.

## Note
*This is a frozen copy of a work-in-progress codebase.* \
*An extended version of this code will be demonstrated in greater detail in a future publication.* \
*Much functionality that is exclusive to that future work has been removed here, but remnants may still remain.*

## Installation
Git is required to download and install the repo. If on Linux or Mac, open Terminal and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/ai-research-students-at-curtin/Traffic-Sign-Damage-Detection-using-Synthesised-Training-Data.git
```

To install the required packages using pip, simply execute the following command (`pip3` may be exchanged with `pip`):
```sh
$ pip3 install -r requirements.txt
```

[comment]: <> (Note that the synthetic dataset SGTSD will need aprox. 10GB and the sample set used for training will be close to 1GB.)


## Usage
Navigate to the `signbreaker` directory. To run the generator, simply execute the following command:
```sh
$ python create_dataset.py
```
The `--output_dir` argument can be used to specify a custom directory for the generated dataset. A complete path is required.

The generator can be configured by modifying the `config.yaml` file. The default values in `config.yaml` represent the values utilised to generated the data used in this paper's  experiments.

Please allow for sufficient storage space when running the dataset generator. With the default config values and the below example inputs, the size of the dataset may be on the order of 1-100 GB depending on the inputs and config.

**Example traffic sign templates** for data generation can be downloaded below. Place these templates into the `signbreaker/Sign_Templates/1_Input` directory. \
[[download]](https://drive.google.com/file/d/1dALYTwtGMGrEXROh8KWBdLzH2_1Jxzmu/view?usp=sharing) | UK | [UK templates](https://www.gov.uk/guidance/traffic-sign-images)  used by Stergiou et al. in their paper. \
[[download]](https://drive.google.com/file/d/19_muDfADDh83zwIndZE3bsfbFh9KrGKD/view?usp=sharing) | GTSDB | classes matched as closely as possible using [UK templates](https://www.gov.uk/guidance/traffic-sign-images). Covers ~31/43 classes. \
[[download]](https://drive.google.com/file/d/1kMAPRSOs9RqAtQu6-fUEn1fqkazIC3Kt/view?usp=sharing) | GTSDB | classes matched using German [Wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany) and [Wikimedia Commons](https://commons.wikimedia.org/wiki/Historic_road_signs_in_Germany#1992%E2%80%932013) images. Covers 43/43 classes.

**Example backgrounds** for data generation can be downloaded below. \
[[download]](https://drive.google.com/file/d/1LvKXLakMttnXL7w4R3dl-dgmkv59cpQK/view?usp=sharing) | Google Images | UK Google Images backgrounds used by Stergiou et al. in their paper, suitable only for creating image patch datasets for recognition tasks. \
[[download]](https://drive.google.com/file/d/1WCfWVruL0_WxnMaYJ-qzQD0cnFO478fh/view?usp=sharing) | GTSDB | 4 GTSDB background images for quick testing. \
[[download]](https://drive.google.com/file/d/1dWkyX9-lGEE59odbthu3zFdZT9ksQ2nS/view?usp=sharing) | GTSDB | all 600 images from the GTSDB training set (warning: if used as is as backgrounds, synthetic data would contain unlabelled signs). \
[[download]](https://drive.google.com/file/d/1navoOiHRhYhrIGgogp1TMoEg3QczvN85/view?usp=sharing) | Various | 1191 images taken from various sources with no visible traffic sign faces. See [here](https://github.com/dsphamgithub/tsda#synthetic-dataset-generation-code) for details.
## Contributors
Kristian Radoš *(Primary)* \
kristianrados40@gmail.com
kristian.rados@student.curtin.edu.au

Seana Dale \
seana.dale@student.curtin.edu.au

Allen Antony \
allenantony2001@gmail.com

Prasanna Asokan \
prasanna.asokan@student.curtin.edu.au

Jack Downes \
jack.downes@postgrad.curtin.edu.au
