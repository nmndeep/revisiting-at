'''Main runnable file for imagenet experiments - tested for multi-gpu training
'''

import sys
sys.path.insert(0,'..')
from math import ceil
import math
import numpy as np
import os
from os import get_terminal_size
from datetime import datetime
import argparse
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import random
import torch as ch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

import argparse
import parserr
from dataset_convnext_like import build_dataset
import torchvision
from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm
import time
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf


import timm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.data.mixup import Mixup
from timm.models import create_model

from autopgd_train_clean import apgd_train
from fgsm_train import fgsm_train, fgsm_attack
from utils_architecture import normalize_model, get_new_model
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_WARNINGS'] = 'off'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: ch.Tensor, target: ch.Tensor) -> ch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        target = target.type(ch.int64)
        nll_loss = -logprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


Section('model', 'model details').params(
    arch=Param(str, default='effnet_b0'),
    pretrained=Param(int, 'is pretrained? (1/0)', default=1),
    ckpt_path=Param(str, 'path to resume model', default=''),
    add_normalization=Param(int, '0 if no normalization, 1 otherwise', default=1),
    not_original=Param(int, 'change effnets? to patch-version', default=0),
    updated=Param(int, 'Make conviso Big? Not in use', default=0),
    model_ema=Param(float, 'Use EMA?', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, 'file to use for training', required=True),
    val_dataset=Param(str, 'file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    seed=Param(int, 'seed for training loader', default=0),
    augmentations=Param(int, 'use fancy augmentations?', default=0)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic', 'cosine']), default='cosine'),
    lr=Param(float, 'learning rate', default=1e-3),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=10),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', default=''),
    log_level=Param(int, '0 if only at end 1 otherwise', default=1),
    save_freq=Param(int, 'save models every nth epoch', default=2),
    addendum=Param(str, 'additional comments?', default=""),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=64),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=0),
    precision=Param(str, 'np precision', default='fp16')
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=0.05),
    epochs=Param(int, 'number of epochs', default=100),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    distributed=Param(int, 'is distributed?', default=0),
    use_blurpool=Param(int, 'use blurpool?', default=0),
    precision=Param(str, 'np precision', default='fp16'),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12357')
)

Section('adv', 'adversarial training options').params(
    attack=Param(str, 'if None standard training', default='none'),
    norm=Param(str, '', default='Linf'),
    eps=Param(float, '', default=4./255),
    n_iter=Param(int, '', default=2),
    verbose=Param(int, '', default=0),
    noise_level=Param(float, '', default=1.),
    skip_projection=Param(int, '', default=0),
    alpha=Param(float, 'step size multiplier', default=1.),
)

Section('misc', 'other parameters').params(
    notes=Param(str, '', default=''),
    use_channel_last=Param(int, 'whether to use channel last memory format', default=1),
)

IMAGENET_MEAN = [c * 1. for c in (0.485, 0.456, 0.406)] #[np.array([0., 0., 0.]), np.array([0.485, 0.456, 0.406])][-1] * 255
IMAGENET_STD = [c * 1. for c in (0.229, 0.224, 0.225)] #[np.array([1., 1., 1.]), np.array([0.229, 0.224, 0.225])][-1] * 255
NONORM_MEAN = np.array([0., 0., 0.])
NONORM_STD = np.array([1., 1., 1.]) * 255
DEFAULT_CROP_RATIO = 224/256

PREC_DICT = {'fp16': np.float16, 'fp32': np.float32}

def sizeof_fmt(num, suffix="Flops"):
    for unit in ["", "Ki", "Mi", "G", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.3f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"



@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cosine_lr(epoch, lr, epochs, lr_peak_epoch):
    if epochs > 100:
        lr_peak_epoch = 20
    else:
        lr_peak_epoch = 10
    if epoch <= lr_peak_epoch:
        xs = [0, lr_peak_epoch]
        ys = [1e-4 * lr, lr]
        return np.interp([epoch], xs, ys)[0]
    else:
        lr_min = 1e-10
        lr_t = lr_min + .5 * (lr - lr_min) * (1 + math.cos(math.pi * (
            epoch - lr_peak_epoch) / (epochs - lr_peak_epoch)))
        return lr_t
    

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)
        

class WrappedModel(nn.Module):
    """ include the generation of adversarial perturbation in the
        forward pass
    """
    def __init__(self, base_model, perturb, verbose=False):
        super().__init__()
        self.base_model = base_model
        self.perturb = perturb
        self.perturb_input = False
        self.verbose = verbose
        #self.mu = mu
        #self.sigma = sigma
        
    def forward(self, x, y=None):
        # TODO: handle varying threat models
        if self.perturb_input:
            assert not y is None
            #print(x.is_contiguous())
            # use eval mode during attack
            self.base_model.eval()
            if self.verbose:
                print('perturb input')
                startt = time.time()
            z = self.perturb(self.base_model, x, y)

            if self.verbose:
                inftime = time.time() - startt
                print(f'inference time={inftime:.5f}')
            #print(z[0].is_contiguous())
            self.base_model.train()
            
            if isinstance(z, (tuple, list)):
                z = z[0]
            return self.base_model(z)
            
        else:
            if self.verbose:
                print('clean inference')
            return self.base_model(x)
            
    def set_perturb(self, mode):
        self.perturb_input = mode


class ImageNetTrainer:
    @param('training.distributed')
    @param('training.eval_only')
    def __init__(self, gpu, distributed, eval_only):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.best_rob_acc = 0.
        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        if not eval_only:
            self.train_loader, self.val_loader, self.mixup_fn = self.create_train_loader()
        # self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()
        

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr,
            'cosine': get_cosine_lr,
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('model.arch')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, arch):

        # Only do weight decay on non-batchnorm parameters
        if 'convnext' in arch or 'resnet' in arch:
            print('manually excluding parameters for weight decay')
            all_params = list(self.model.named_parameters())
            excluded_params = ['bn', '.bias'] #'.norm', '.bias'
            if arch in ['timm_convnext_tiny_batchnorm', 'timm_convnext_tiny_batchnorm_relu']:
                # timm convnext uses different naming than resnet
                excluded_params.append('.norm.')
            if arch in ['timm_resnet50_dw_patch-stem_gelu_stages-3393_convnext-bn_fewer-act-norm_ln',
                'timm_resnet50_dw_patch-stem_gelu_stages-3393_convnext-bn_fewer-act-norm_ln_ds-sep',
                'timm_resnet50_dw_patch-stem_gelu_stages-3393_convnext-bn_fewer-act-norm_ln_ds-sep_bias',
                'timm_reimplemented_convnext_tiny']:
                # in case LN is used instead of original BN and the naming is not changed
                excluded_params.remove('bn')
            print('excluded params=', ', '.join(excluded_params))
            
            bn_params = [v for k, v in all_params if any([c in k for c in excluded_params])] #('bn' in k) #or k.endswith('.bias')
            bn_keys = [k for k, v in all_params if any([c in k for c in excluded_params])]
            other_params = [v for k, v in all_params if not any([c in k for c in excluded_params])]  #not ('bn' in k) #or k.endswith('.bias')

        else:
            print('automatically exclude bn and bias from weight decay')
            bn_params = []
            bn_keys = []
            other_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"): #or name in no_weight_decay_list
                    bn_keys.append(name)
                    bn_params.append(param)
                else:
                    other_params.append(param)
            #print(', '.join(bn_keys))
        
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]


        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        else:
            self.optimizer = ch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        if self.mixup_fn is None:
            self.loss = ch.nn.CrossEntropyLoss()
        else:
        # # smoothing is handled with mixup label transform
            # self.loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.loss = SoftTargetCrossEntropy()

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('training.distributed')
    @param('training.label_smoothing')
    @param('data.in_memory')
    @param('data.seed')
    @param('data.augmentations')
    @param('training.precision')
    @param('misc.use_channel_last')
    @param('dist.world_size')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            distributed, label_smoothing, in_memory, seed, augmentations, precision,
                            use_channel_last, world_size):
        ch.manual_seed(seed)
        

        if augmentations:
            args = parserr.Arguments_augment()

        else:
            args = parserr.Arguments_No_augment()

        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        if False:
            args.dist_eval = False
            dataset_val = None
        else:
            dataset_val, _ = build_dataset(is_train=False, args=args)

        num_tasks = world_size
        global_rank = self.gpu #utils.get_rank()

        sampler_train = ch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=seed,
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = ch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = ch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = ch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if dataset_val is not None:
            data_loader_val = ch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(1.5 * batch_size),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        else:
            data_loader_val = None

        mixup_fn = None
        mixup_active = (args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None) and augmentations
        if mixup_active:
            print("Mixup is activated!")
            print(f"Using label smoothing:{label_smoothing}")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=label_smoothing, num_classes=args.nb_classes)

        return data_loader_train, data_loader_val, mixup_fn

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    @param('validation.precision')
    @param('training.distributed')
    @param('misc.use_channel_last')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution, precision, distributed, use_channel_last
                          ):
        '''Validations stats are not computed during training - to save time and compute'''
       
        loader = None
        return loader

    @param('training.epochs')
    @param('logging.log_level')
    @param('logging.save_freq')
    @param('model.ckpt_path')
    @param('adv.attack')

    def train(self, epochs, log_level, save_freq, ckpt_path, attack):
#         vall, nums = self.single_val()
#         if log_level > 0:
#             val_dict = {
#                 'Validation acc': vall.item(),
#                 'points': nums
#             }
#             if self.gpu == 0:
#                 self.log(val_dict)

        for epoch in range(epochs):
            #print(f'epoch {epoch}')
            res = self.get_resolution(epoch)
            try:
                self.decoder.output_size = (res, res)
            except:
                pass
            train_loss = self.train_loop(epoch)

            if log_level > 0:
                extra_dict = {
                    'train_loss': train_loss.item(),
                    'epoch': epoch
                }

                self.eval_and_log(extra_dict)
                
            if train_loss.isnan():
                sys.exit()
            
            if attack == 'none':
                save_freq = 1
            
            self.eval_and_log({'epoch': epoch})
            if (self.gpu == 0 and epoch % save_freq == 0) or (self.gpu == 0 and epoch == epochs - 1):
                ch.save(self.model.state_dict(), self.log_folder / f'weights_{epoch}.pt')
                if self.model_ema is not None:
                    ch.save(timm.utils.model.get_state_dict(self.model_ema), self.log_folder / f'weights_ema_{epoch}.pt')
                    if epoch % 5 == 0 or epoch == epochs-1:
                        ch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_scaler_state_dict': self.scaler.state_dict(),
                        'epoch': epoch,
                        'state_dict_ema':timm.utils.model.get_state_dict(self.model_ema)
                        }, self.log_folder / f'full_model_{epoch}.pth')
                else:
                    if epoch % 5 == 0 or epoch == epochs-1:
                        ch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss_scaler_state_dict': self.scaler.state_dict(),
                        'epoch': epoch,
                        }, self.log_folder / f'full_model_{epoch}.pth')

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = 0 #self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict({
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'top_1': stats,
                'top_5': stats,
                'val_time': 0
            }, **extra_dict))

        return stats


    @param('model.arch')
    @param('model.pretrained')
    @param('model.not_original')
    @param('model.updated')
    @param('model.model_ema')
    @param('training.distributed')
    @param('training.use_blurpool')
    @param('model.ckpt_path')
    @param('model.add_normalization')
    @param('adv.attack')
    @param('adv.norm')
    @param('adv.eps')
    @param('adv.n_iter')
    @param('adv.verbose')
    @param('misc.use_channel_last')
    @param('adv.alpha')
    @param('adv.noise_level')
    @param('adv.skip_projection')
    def create_model_and_scaler(self, arch, pretrained, not_original, updated, model_ema, distributed, use_blurpool,
        ckpt_path, add_normalization, attack, norm, eps, n_iter, verbose,
        use_channel_last, alpha, noise_level, skip_projection):
        scaler = GradScaler()
        if not arch.startswith('timm_'):
            model = get_new_model(arch, pretrained=bool(pretrained), not_original=bool(not_original), updated=bool(updated))
        else:
            try:
                model = create_model(arch.replace('timm_', ''), pretrained=pretrained)
                #model.drop_path_rate = .1
            except:
                model = get_new_model(arch.replace('timm_', ''))
        verbose = verbose == 1
        
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)
        
        if use_channel_last:
            print('using channel last memory format')
            model = model.to(memory_format=ch.channels_last)
        else:
            print('not using channel last memory format')
      
        if add_normalization:
            print('add normalization layer')
            model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)

        if attack in ['apgd', 'fgsm']:
            print('using input perturbation layer')
            if attack == 'apgd':
                attack = partial(apgd_train, norm=norm, eps=eps,
                    n_iter=n_iter, verbose=verbose, mixup=self.mixup_fn)
            elif attack == 'fgsm':
                attack = partial(fgsm_train, eps=eps,
                    use_rs=True,
                    alpha=alpha,
                    noise_level=noise_level,
                    skip_projection=skip_projection == 1
                    )
            print(attack)
            model = WrappedModel(model, attack, verbose=verbose)
        
        if self.gpu == 0:
            print(model)

        if not ckpt_path == '':
            ckpt = ch.load(ckpt_path, map_location='cpu')
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            try:
                model.load_state_dict(ckpt)
                print('standard loading')

            except:
                try:
                    # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
                    model.load_state_dict(ckpt)
                    print('loaded from clean model')
                except:
                    ckpt = {k.replace('base_module', ''): v for k, v in ckpt.items()}
                    ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
                    model.load_state_dict(ckpt)
                    print('loaded')
        
        model = model.to(self.gpu)
        if bool(model_ema):
            print('Using EMA with decay 0.9999')
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            self.model_ema = timm.utils.ModelEmaV2(model, decay=0.9999, device='cpu')
        else:
            self.model_ema = None

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu]) #, find_unused_parameters=True)

        return model, scaler

    @param('validation.lr_tta')
    @param('adv.attack')
    @param('dist.world_size')
    def single_val(self, lr_tta, attack, world_size):
        model = self.model
        model.eval()
        show_once = True
        acc = 0.
        accs = []
        n = 0.
        ns = []
        best_test_rob = 0.

        with autocast(enabled=True):
            for idx, (images, target) in enumerate(tqdm(self.val_loader)):
                # if show_once:
                #     print(images.shape, images.max(), images.min())
                #     show_once = False
                    
                images = images.contiguous().cuda(self.gpu, non_blocking=True)
                target = target.contiguous().cuda(self.gpu, non_blocking=True)
                output = self.model(images)
                if lr_tta:
                    output += self.model(ch.flip(images, dims=[3]))

                # for k in ['top_1', 'top_5']:
                #     self.val_meters[k](output, target)
                    
                acc += (output.max(1)[1] == target).sum()
                n += target.shape[0]
                # loss_val = self.loss(output, target)  #####. remove this comment
                # self.val_meters['loss'](loss_val)
                if idx >= 200:
                    break
        accs.append(acc)
        ns = n*world_size
        print(f'clean accuracy={acc / n:.2%}')
        # stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        # if stats['top_1'] > self.best_rob_acc:
        #     self.best_rob_acc = stats['top_1']
            # if self.gpu == 0:
            #     ch.save(self.model.state_dict(), self.log_folder / 'best_adv_weights.pt')
        # [meter.reset() for meter in self.val_meters.values()]
        return ch.stack(accs)/ns, ns

    @param('logging.log_level')
    @param('adv.attack')
    @param('training.distributed')
    def train_loop(self, epoch, log_level, attack, distributed):
        model = self.model
        model.train()
        losses = []
        show_once = True
        perturb = attack != 'none'
        if perturb:
            if distributed:
                model.module.set_perturb(True)
            else:
                model.set_perturb(True)

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target) in enumerate(iterator):
            images = images.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
            # print(images.size(), target.size())
            if self.mixup_fn is not None:
                images, target = self.mixup_fn(images, target)
                
            if show_once:
                # print(images.shape, images.max(), images.min())
                show_once = False
        
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lrs[ix]
                

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                if not perturb:
                    output = self.model(images)
                else:
                    output = self.model(images, target) # TODO: check the effect of .contiguous() for other models
                loss_train = self.loss(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            ### Training end
            if self.model_ema is not None:
                self.model_ema.update(model)
            
            #ch.cuda.synchronize()

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss_train.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

            
        if perturb:
            if distributed:
                model.module.set_perturb(False)
            else:
                model.set_perturb(False)
            
        return ch.stack(losses).mean()

    @param('validation.lr_tta')
    @param('adv.attack')
    def val_loop(self, lr_tta, attack):

        model = self.model
        model.eval()
        show_once = True
        acc = 0.
        best_test_rob = 0
        # with ch.no_grad():
        with autocast(enabled=True):
            for idx, (images, target) in enumerate(tqdm(self.val_loader)):
                # if show_once:
                #     print(images.shape, images.max(), images.min())
                #     show_once = False
                    
                images = images.contiguous()
                target = target.contiguous()
                # if attack != 'none':
                #     x_adv = fgsm_attack(model, images, target, eps=4./255.)
                #     output = self.model(x_adv)
                #     if lr_tta:
                #         output += self.model(ch.flip(x_adv, dims=[3]))
                # else:
                output = self.model(images)
                if lr_tta:
                    output += self.model(ch.flip(images, dims=[3]))
                # if lr_tta:
                #     output += self.model(ch.flip(x_adv, dims=[3]))
                for k in ['top_1', 'top_5']:
                    self.val_meters[k](output, target)
                    
                acc += (output.max(1)[1] == target).sum()

                loss_val = self.loss(output, target)
                self.val_meters['loss'](loss_val)
                if idx >= 50:
                    break
        stats = {k: m.compute().item() for k, m in self.val_meters.items()}

        if stats['top_1'] > self.best_rob_acc:
            self.best_rob_acc = stats['top_1']
            if self.gpu == 0:
                ch.save(self.model.state_dict(), self.log_folder / 'best_adv_weights.pt')
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    @param('logging.folder')
    @param('model.arch')
    @param('adv.attack')
    @param('model.updated')
    @param('model.not_original')
    @param('logging.addendum')
    @param('data.augmentations')
    @param('model.pretrained')
    def initialize_logger(self, folder, arch, attack, updated, not_original, addendum, augmentations, pretrained):
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

        if self.gpu == 0:
            #folder = (Path(folder) / str(self.uid)).absolute()
            runname = f'model_{str(datetime.now())[:-7]}_{arch}_upd_{updated}_not_orig_{not_original}_pre_{pretrained}_aug_{augmentations}'
            if attack != 'none':
                runname += f'_adv_{addendum}'
            else:
                runname += f'_clean_{addendum}'
            folder = (Path(folder) / runname).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }
            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        cur_time = time.time()
        try:
            with open(self.log_folder / 'log', 'a+') as fd:
                fd.write(json.dumps({
                    'timestamp': cur_time,
                    'relative_time': cur_time - self.start_time,
                    **content
                }) + '\n')
                fd.flush()
        except:
            with open(self.log_folder / 'log', 'a+') as fd:
                fd.write(content + '\n')
                fd.flush()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()
        if distributed:
            trainer.cleanup_distributed()

# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    ImageNetTrainer.launch_from_args()
