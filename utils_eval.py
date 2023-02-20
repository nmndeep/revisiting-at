import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import math
import os
import json

# from data import load_cifar10c
from models_new import l_models_all, l_models_imagenet, l_models_cifar100,\
    l_models_imagenet100, l_models_mnist
from utils import load_anymodel, load_anymodel_imagenet, clean_accuracy,\
    load_anymodel_cifar100, load_anymodel_imagenet100, load_anymodel_mnist
try:
    from other_utils import L1_norm, L2_norm, Linf_norm, Logger, L0_norm
except ImportError:
    from autoattack.other_utils import L1_norm, L2_norm, Logger, L0_norm
from apgd_mask import criterion_dict


cls = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
    

class CalibratedModel(nn.Module):
    def __init__(self, model, temp):
        super().__init__()
        assert not model.training
        self.model = model
        assert temp > 0.
        self.temp = temp
        
    def forward(self, x):
        return self.model(x) / self.temp


def Linf_norm():
    raise NotImplementedError('Linf_norm to be added.')


def get_acc_cifar10c(model, n_ex=10000, severities=[5],
    corruptions=('shot_noise', 'motion_blur', 'snow', 'pixelate',
    'gaussian_noise', 'defocus_blur', 'brightness', 'fog', 'zoom_blur',
    'frost', 'glass_blur', 'impulse_noise', 'contrast',
    'jpeg_compression', 'elastic_transform'), bs=250):
    l_acc = []
    acc_dets = {}
    
    for s in severities:
        x, y = load_cifar10c(n_ex, severity=s, corruptions=corruptions)
        x = x.contiguous()
        print(x.shape)
        with torch.no_grad():
            acc = 0.
            n_batches = math.ceil(x.shape[0] / bs)
            for counter in range(n_batches):
                output = model(x[counter * bs:(counter + 1) * bs].cuda())
                acc += (output.cpu().max(dim=1)[1] == y[counter * bs:(counter + 1) * bs]).sum()
        l_acc.append(acc / x.shape[0])
        acc_dets[str(s)] = acc / x.shape[0]
        print('sev={}, clean accuracy={:.2%}'.format(s, acc / x.shape[0]))

    return acc / x.shape[0], acc_dets

def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    print(adv.max().item() - 1., adv.min().item())
    
    return str_det

def get_cifar10_class(lab):
    return cls[lab]

def get_imagenet_class(lab):
    if torch.is_tensor(lab):
        lab = lab.item()
    with open('./imagenet_classes.json') as json_file:
        class_dict = json.load(json_file)
    return class_dict[str(lab)][1]

def get_class(args, cl=None):
    if cl is None:
        cl = args.target_class
    if args.dataset == 'cifar10':
        return get_cifar10_class(cl)
    elif args.dataset == 'imagenet':
        return get_imagenet_class(cl)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_anymodel_datasets(args):
    fts_idx = [int(c) for c in args.fts_idx.split(' ')]
    if args.dataset == 'cifar10':
        l_models = [l_models_all[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel(l_models[0], args.model_dir)
        model.eval()
    elif args.dataset == 'imagenet':
        l_models = [l_models_imagenet[c] for c in fts_idx]
        print(l_models)
        kwargs = {}
        if (l_models[0][0].startswith('DeiT')
            #and 'convblock' not in l_models[0][0]
            or l_models[0][0].startswith('ViT')
            ):
            kwargs = {'img_size': args.img_size}
        model = load_anymodel_imagenet(l_models[0], **kwargs)
        #sys.exit()
        '''with torch.no_grad():
            acc = clean_accuracy(model, x, y, batch_size=25)
        print('clean accuracy: {:.1%}'.format(acc))'''
    elif args.dataset == 'cifar100':
        l_models = [l_models_cifar100[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel_cifar100(l_models[0])
        model.eval()
    elif args.dataset == 'imagenet100':
        l_models = [l_models_imagenet100[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel_imagenet100(l_models[0])
        model.eval()
    elif args.dataset == 'mnist':
        l_models = [l_models_mnist[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel_mnist(l_models[0])
        model.eval()
    return model, l_models


def attack_group(norm, suffix=''):
    if norm in ['Linf', 'L2', 'L1']:
        return 'aa' + suffix
    else:
        return 'nonlpattacks'


def get_norm(z, norm):
    if norm == 'Linf':
        return Linf_norm(z)
    elif norm == 'L2':
        return L2_norm(z)
    elif norm == 'L1':
        return L1_norm(z)
    elif norm == 'L0':
        return L0_norm(z)


def get_logits(model, x_test, bs=1000, device=None, n_cls=10):
    if device is None:
        device = x_test.device
    n_batches = math.ceil(x_test.shape[0] / bs)
    logits = torch.zeros([x_test.shape[0], n_cls], device=device)
    #l_logits = []
    
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
            output = model(x_curr)
            #l_logits.append(output.detach())
            logits[counter * bs:(counter + 1) * bs] += output.detach()
    
    return logits


def get_wc_acc(model, xs, y, bs=1000, device=None, eot_test=1, logger=None,
    loss=None, n_cls=10):
    if device is None:
        device = x.device
    if logger is None:
        logger = Logger(None)
    if not loss is None:
        criterion_indiv = criterion_dict[loss]
    y = y.to(device)
    acc = torch.ones_like(y, device=device).float()
    x_adv = xs[0].clone()
    loss_best = -1. * float('inf') * torch.ones(y.shape[0], device=device)
    
    for x in xs:
        logits = get_logits(model, x, bs=bs, device=device, n_cls=n_cls)
        loss_curr = criterion_indiv(logits, y)
        pred_curr = logits.max(1)[1] == y
        ind = ~pred_curr * (loss_curr > loss_best) # misclassified points with higher loss
        x_adv[ind] = x[ind].clone()
        acc *= pred_curr
        ind = (acc == 1.) * (loss_curr > loss_best) # for robust points track highest loss
        x_adv[ind] = x[ind].clone()
        logger.log(f'[rob acc] cum={acc.mean():.1%} curr={pred_curr.float().mean():.1%}')
    
    print(torch.nonzero(acc).squeeze())
    
    return acc.mean(), x_adv


def get_patchsize(dataset, modelname):
    if dataset == 'imagenet':
        if modelname in ['ConvMixer_1024_20_nat', 'ConvMixer_1024_20_eps4_best']:
            return 14
        else:
            return 16
    else:
        return 8
