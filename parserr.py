'''Helper script to generate the parameter values for augmentations
'''

import argparse

def str2bool(v):

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Arguments_augment():
    def __init__(self):
        self.color_jitter = 0.4
        self.aa = 'rand-m9-mstd0.5-inc1'
        self.train_interpolation = 'bicubic'
        self.crop_pct = None
        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1
        self.resplit = False
        self.mixup = 0.8
        self.cutmix = 1.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'
        self.nb_classes = 1000
        self.input_size = 224
        self.data_set = 'IMNET'
        self.dist_eval = True
        self.hflip = 0.5
        self.vflip = 0.0
        self.scale = [0.08, 1.0]
        self.ratio = [3./4., 4./3.]



class Arguments_No_augment():
    def __init__(self):
        self.color_jitter = 0.0
        self.aa = None #'rand-m9-mstd0.5-inc1'
        self.train_interpolation = 'bicubic'
        self.crop_pct = None
        self.reprob = 0.0
        self.remode = None #'pixel'
        self.recount = 0
        self.resplit = False
        self.mixup = 0.0
        self.cutmix = 0.
        self.cutmix_minmax = None
        self.mixup_prob = 0.0
        self.mixup_switch_prob = 0. 
        self.mixup_mode = None
        self.nb_classes = 1000
        self.data_set = 'IMNET'
        self.input_size = 224
        self.dist_eval = True
        self.hflip = 0.0
        self.vflip = 0.0
        self.scale = [0.08, 1.0]
        self.ratio = [3./4., 4./3.]
