# revisiting-at
Code for training and evaluating models robust to $\ell_p$-norm adversaries for ImageNet, and finetuning to other datasets

## Code
Requirements (specific versions tested on): </br>
`fastargs-1.2.0` `autoattack-0.1` `pytorch-1.13.1` `torchvision-0.14.1` `robustbench-1.1` `timm-0.8.0.dev0`

#### Training
The bash script in `run_train.sh` trains the model `model.arch`. For clean training: `adv.attack None` and for adversarial training set `adv.attack apgd`.</br>
For the standard setting as in paper (full augmentations) set `data.augmentations 1`, `model.model_ema 1` and `training.label_smoothing 1`.</br>
To train models with Convolution-Stem (CvSt) set `model.not_original 1`. </br>
The code does standard APGD adversarial trainining. </br>The file `utils_architecture.py` has model definitions for the new `CvSt` models.

#### Evaluating a model
The file `runner_aa_eval` runs `AutoAttack`(AA). Passing `fullaa 1` runs complete AA whereas `fullaa 0` runs the first two attacks in AA.</br>
The best checkpoints for the models in the paper will be added soon
