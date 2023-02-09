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

## Publication

If you find EczemaNet2 useful to your research, please cite our [JID'22 paper](https://www.medrxiv.org/content/10.1101/2022.11.05.22281951v1.full.pdf):
```
@article {EczemaNet2,
	author = {Attar, Rahman and Hurault, Guillem and Wang, Zihao and Mokhtari, Ricardo and Pan, Kevin and Olabi, Bayanne and Earp, Eleanor and Steele, Lloyd and Williams, Hywel C. and Tanaka, Reiko J.},
	title = {Reliable detection of eczema areas for fully automated assessment of eczema severity from digital camera images},
	year = {2022},
	journal = {JID Innovations}
}
```

The original EczemaNet research can be found [here](https://github.com/Tanaka-Group/EczemaNet).

## License
This open source version of EczemaNet2 is licensed under the GPLv3 license, which can be found in the [LICENSE](/LICENSE) file.

A closed source version of EczemaNet is also available without the restrictions of the GPLv3 license with a software usage agreement from Imperial College London. For more information, please contact Diana Yin <d.yin@imperial.ac.uk>.

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