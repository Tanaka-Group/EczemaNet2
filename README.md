# EczemaNet2: Reliable detection of eczema areas for fully automated assessment of eczema severity from digital camera images

Assessing the severity of eczema in clinical research requires face-to-face skin examination
by trained staff. Such approaches are resource-intensive for participants and staff,
challenging during pandemics, and prone to inter- and intra-observer variation. Computer
vision algorithms have been proposed to automate the assessment of eczema severity using
digital camera images. However, they often require human intervention to detect eczema
lesions and cannot automatically assess eczema severity from real-world images in an endto-end pipeline.

We developed a new model to detect eczema lesions from images using data augmentation
and pixel-level segmentation of eczema lesions on 1345 images provided by dermatologists.
We evaluated the quality of the obtained segmentation compared to that of the clinicians,
the robustness to varying imaging conditions encountered in real-life images, such as
lighting, focus, and blur and the performance of downstream severity prediction when using
the detected eczema lesions. The quality and robustness of eczema lesion detection
increased by approximately 25% and 40%, respectively, compared to our previous eczema
detection model. The performance of the downstream severity prediction remained
unchanged.

## Setup

EczemaNet2 are build to work together with Keras and TensorFlow Keras frameworks, the major requirements are listed below:
* Python 3.7
* h5py < 3.0.0
* tensorflow-gpu 1.15
* keras 2.3.0

We recommend to setup EczemaNet2 with [conda](https://anaconda.org/anaconda/conda) as it provides better dependency management and isolation:
```
# Create the environment with conda from the environment.yml file
conda env create -f environment.yml

# Activate the new environment
source activate YOUR_NEW_ENV
```

Alternatively, the dependencies can be manually installed from `environment.yml` with pip.

## Data

For privacy concerns, the Softened Water Eczema Trial (SWET) dataset that we used in EczemaNet2 is not shareable. You are welcome to train the model with your data source. However, if you do need to access SWET, please contact [Dr Reiko Tanaka](mailto:r.tanaka@imperial.ac.uk) for more information.

## Training

To train EczemaNet2, execute `src/train_batch.py` by giving the following parameters as inputs:
* The type of segmentation (either `SKIN` or `AD` (stands for Atopic Dermatitis)).
* The directory of the training set.
* The preferred prefix name to identify your model.

Usage:
```
python /PROJ_DIR/src/train_batch.py --seg_type SKIN --train_dir /PROJ_DIR/data/training_set --prefix base
```

## Evaluation

* `src/eval.py`: evaluates the cropping quality of skin segmentation in identifying skin regions, and AD segmentation in identifying AD regions.
* `src/eval_of_ad_identification.py`: evaluates the cropping quality of both skin and AD segmentation in identifying AD regions.
* `src/eval_of_robustness.py`: evaluates the robustness for skin and AD segmentation.

Usage:
```
python /PROJ_DIR/src/eval.py --seg_type AD --suffix base --model_dir /PATH_TO_YOUR_MODEL
```

## Publication

As you use EczemaNet2 for your exciting discoveries, please cite our [JID'22 paper](https://www.medrxiv.org/content/10.1101/2022.11.05.22281951v1.full.pdf):
```
@article {EczemaNet2,
	author = {Attar, Rahman and Hurault, Guillem and Wang, Zihao and Mokhtari, Ricardo and Pan, Kevin and Olabi, Bayanne and Earp, Eleanor and Steele, Lloyd and Williams, Hywel C. and Tanaka, Reiko J.},
	title = {Reliable detection of eczema areas for fully automated assessment of eczema severity from digital camera images},
	year = {2022},
	journal = {JID Innovations}
}
```
You may also find the original EczemaNet research useful, which can be found [here](https://github.com/Tanaka-Group/EczemaNet).

## License

This open source version of EczemaNet2 is licensed under the GPLv3 license, which can be found in the [LICENSE](/LICENSE) file.

A closed source version of EczemaNet is also available without the restrictions of the GPLv3 license with a software usage agreement from Imperial College London. For more information, please contact [Diana Yin](mailto:d.yin@imperial.ac.uk).

```
EczemaNet2: Reliable detection of eczema areas for fully automated
assessment of eczema severity from digital camera images
Copyright (C) 2022 Tanaka Group <r.tanaka@imperial.ac.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

```