# revisiting-at
Code for training and evaluating models robust to $\ell_p$-norm adversaries for ImageNet, and finetuning to other datasets.

## Code
Requirements (specific versions tested on): </br>
`fastargs-1.2.0` `autoattack-0.1` `pytorch-1.13.1` `torchvision-0.14.1` `robustbench-1.1` `timm-0.8.0.dev0`

#### Training
The bash script in `run_train.sh` trains the model `model.arch`. For clean training: `adv.attack None` and for adversarial training set `adv.attack apgd`.</br>
For the standard setting as in paper (heavy augmentations) set `data.augmentations 1`, `model.model_ema 1` and `training.label_smoothing 1`.</br>
To train models with Convolution-Stem (CvSt) set `model.not_original 1`. </br>
The code does standard APGD adversarial trainining. </br>The file `utils_architecture.py` has model definitions for the new `CvSt` models.

#### Evaluating a model
The file `runner_aa_eval` runs `AutoAttack`(AA). Passing `fullaa 1` runs complete AA whereas `fullaa 0` runs the first two attacks in AA.</br>


##### Checkpoints - ImageNet $\ell_{\infty} = 4/255$ robust models.
The link location includes the clean model (the one used as init for AT), the robust model, and the `full-AA` log for $\ell_{\infty}, \ell_2$ and $\ell_1$ attacks.</br>
| Model-Name           | epochs | Clean acc. | $\ell_{\infty}$ acc.|  Location (Link) |
| :---                 | :------: | :------:   |:------: | :------:    |   
| ConvNext-iso-CvSt    | 300 | 70.2  | 45.9 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/HpNbkLTNTBiaeo8)|
| ViT-S-CvSt           | 300 | 72.5  | 48.1 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/agtDw3D7QXbDCmw)|
| ConvNext-T-CvSt      | 300 | 72.7  | 49.5 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/BFLoMrMdn8iBk7Y)|
| ViT-M-CvSt           | 50 | 72.4  | 48.8 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/q2mkEYtq5Zjpa4e)|
| ConvNext-S-CvSt      | 50 | 74.4  | 52.4 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/m3bAwNg4CJY4jrp)|
| ViT-B-CvSt           | 300 | 76.3  | 54.7 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/SbN5AJAicdZJXyr)|
| ConvNext-B-CvSt      | 300 | 75.9  | 56.1 |   [Link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/RQBEXagC7R7XweX)|

