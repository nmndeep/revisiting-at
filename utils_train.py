import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import copy
import random
import math
import time
from collections import OrderedDict
from typing import Tuple
from torch import Tensor
import os

try:
    from autopgd_pt import L1_projection
except ImportError:
    from autoattack.autopgd_base import L1_projection

class FGSMAttack():
    def __init__(self, eps=8. / 255., step_size=None, loss=None):
        self.eps = eps
        self.step_size = step_size if not step_size is None else eps
        self.loss = loss
    
    def perturb(self, model, x, y, random_start=False):
        assert not self.loss is None
        if random_start:
            t = (x + (2. * torch.rand_like(x) - 1.) * self.eps).clamp(0., 1.)
        else:
            t = x.clone()

        t.requires_grad = True
        output = model(t)
        loss = self.loss(output, y)
        grad = torch.autograd.grad(loss, t)[0]

        x_adv = x + grad.detach().sign() * self.step_size
        return torch.min(torch.max(x_adv, x - self.eps), x + self.eps).clamp(0., 1.)

class PGDAttack():
    def __init__(self, eps=8. / 255., step_size=None, loss=None, n_iter=10,
        norm='Linf', verbose=False):
        self.eps = eps
        self.step_size = step_size if not step_size is None else eps / n_iter * 1.5
        self.loss = loss
        self.n_iter = n_iter
        self.norm = norm
        self.verbose = verbose
    
    def perturb(self, model, x, y, random_start=False, return_acc=False):
        assert not self.loss is None
        if random_start:
            x_adv = (x + (2. * torch.rand_like(x) - 1.) * self.eps).clamp(0., 1.)
        else:
            x_adv = x.clone()

        n_fts = x.shape[1] * x.shape[2] * x.shape[3]
        x_best, loss_best = x_adv.clone(), torch.zeros_like(y).float()
        acc = torch.ones_like(y).detach().float()
        
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = self.loss(output, y, reduction='none')
        ind = loss > loss_best
        x_best[ind] = x_adv[ind].clone().detach()
        loss_best[ind] = loss[ind].clone().detach()
        acc[ind] = (output.max(dim=1)[1] == y).float()[ind].clone().detach()
        grad = torch.autograd.grad(loss.mean(), x_adv)[0]
        
        for it in range(self.n_iter):
            if self.norm == 'Linf':
                x_adv = x_adv.detach() + grad.detach().sign() * self.step_size
                x_adv = torch.min(torch.max(x_adv, x - self.eps), x + self.eps).clamp(0., 1.)
        
            elif self.norm == 'L2':
                x_adv = x_adv.detach() + grad.detach() / (grad.detach() ** 2
                    ).sum(dim=(1, 2, 3), keepdim=True).sqrt() * self.step_size
                delta_l2norm = ((x_adv - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                x_adv = (x + (x_adv - x) * torch.min(torch.ones_like(delta_l2norm),
                    self.eps * torch.ones_like(delta_l2norm) / delta_l2norm)).clamp(0., 1.)
    
            elif self.norm == 'L1':
                grad = grad.detach()
                grad_topk = grad.abs().view(grad.shape[0], -1).topk(
                    k=max(int(.1 * n_fts), 1), dim=-1)[0][:, -1].view(grad.shape[0],
                    *[1]*(len(grad.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv = x_adv.detach() + self.step_size * sparsegrad / (sparsegrad.abs().view(
                    x.shape[0], -1).sum(dim=-1).view(-1, 1, 1, 1) + 1e-10)
                delta_temp = L1_projection(x, x_adv - x, self.eps)
                x_adv += delta_temp
        
            x_adv.requires_grad = True
            output = model(x_adv)
            loss = self.loss(output, y, reduction='none')
            ind = loss > loss_best
            x_best[ind] = x_adv[ind].clone().detach()
            loss_best[ind] = loss[ind].clone().detach()
            acc[ind] = (output.max(dim=1)[1] == y).float()[ind].clone().detach()
            grad = torch.autograd.grad(loss.mean(), x_adv)[0]
            
            if self.verbose:
                print('[{}] it={} loss={:.5f} acc={:.1%}'.format(self.norm, it, loss_best.mean().item(),
                    (output.max(dim=1)[1] == y).cpu().float().mean()))
        
        if not return_acc:
            return x_best.detach()
        else:
            return x_best.detach(), acc

class MSDAttack():
    def __init__(self, eps, step_size=None, loss=None, n_iter=10):
        self.eps = eps
        self.step_size = step_size if not step_size is None else [eps / n_iter * 1.25 for eps in self.eps]
        self.loss = loss
        self.n_iter = n_iter

    def perturb(self, model, x, y, random_start=False):
        assert not self.loss is None
        if random_start:
            x_adv = (x + (2. * torch.rand_like(x) - 1.) * self.eps).clamp(0., 1.)
        else:
            x_adv = x.clone()

        n_fts = x.shape[1] * x.shape[2] * x.shape[3]
        x_best, loss_best = x_adv.clone(), torch.zeros_like(y).float()
        #x_adv.requires_grad = True
        
        for it in range(self.n_iter):
            x_adv.requires_grad = True
            #with torch.enable_grad()
            output = model(x_adv)
            loss = self.loss(output, y, reduction='none')
            avgloss = loss.mean()
            grad = torch.autograd.grad(avgloss, x_adv)[0]
            ind = loss > loss_best
            x_best[ind] = x_adv[ind].clone().detach()
            loss_best[ind] = loss[ind].clone().detach()
            #grad = torch.autograd.grad(loss.mean(), x_adv)[0].detach()
    
            x_adv_linf = x_adv.detach() + grad.detach().sign() * self.step_size[0]
            x_adv_linf = torch.min(torch.max(x_adv_linf, x - self.eps[0]), x + self.eps[0]).clamp(0., 1.)

            x_adv_l2 = x_adv.detach() + grad.detach() / (grad.detach() ** 2
                ).sum(dim=(1, 2, 3), keepdim=True).sqrt() * self.step_size[1]
            delta_l2norm = ((x_adv_l2 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
            x_adv_l2 = (x + (x_adv_l2 - x) * torch.min(torch.ones_like(delta_l2norm),
                self.eps[1] * torch.ones_like(delta_l2norm) / delta_l2norm)).clamp(0., 1.)

            grad = grad.detach()
            grad_topk = grad.abs().view(grad.shape[0], -1).topk(
                k=max(int(.1 * n_fts), 1), dim=-1)[0][:, -1].view(grad.shape[0],
                *[1]*(len(grad.shape) - 1))
            sparsegrad = grad * (grad.abs() >= grad_topk).float()
            x_adv_l1 = x_adv.detach() + self.step_size[2] * sparsegrad / (sparsegrad.abs().view(
                x.shape[0], -1).sum(dim=-1).view(-1, 1, 1, 1) + 1e-10)
            delta_temp = L1_projection(x, x_adv_l1 - x, self.eps[2])
            x_adv_l1 += delta_temp

            '''x_adv_linf.requires_grad_()
            x_adv_l2.requires_grad_()
            x_adv_l1.reuiqres_grad_()'''
            l_x_adv = [x_adv_linf, x_adv_l2, x_adv_l1]
            l_output = [model(x_adv_linf), model(x_adv_l2), model(x_adv_l1)]
            l_loss = [self.loss(c, y, reduction='none') for c in l_output]
            #l_avgloss = [c.mean() for c in l_loss]
            val_max, ind_max = torch.max(torch.stack(l_loss, dim=1), dim=1)
            #x_adv, loss = l_x_adv[ind_max], l_loss[ind_max]
            x_adv = x_adv_linf.clone()
            x_adv[ind_max == 1] = x_adv_l2[ind_max == 1].clone()
            x_adv[ind_max == 2] = x_adv_l1[ind_max == 2].clone()
            #print('it={} - best norm=({}, {}, {}) - loss={}'.format(it + 1, (ind_max == 0).sum(),
            #    (ind_max == 1).sum(), (ind_max == 2).sum(), loss_best.mean().item()))
        
        return x_best.detach()

class MultiPGDAttack():
    def __init__(self, eps, step_size=None, loss=None, n_iter=[10, 10, 10], use_miscl=False,
        l_norms=None):
        self.eps = eps
        self.step_size = step_size if not step_size is None else [eps / n_iter * 1.5 for eps in self.eps]
        self.loss = loss
        self.n_iter = n_iter
        self.indiv_adversary = PGDAttack(eps[0], loss=loss)
        self.use_miscl = use_miscl
        self.l_norms = l_norms if not l_norms is None else ['Linf', 'L2', 'L1']
    
    def perturb(self, model, x, y, random_start=False, return_acc=False):
        #assert not self.loss is None
        l_x_adv = []
        l_acc = []
        for i, norm in enumerate(self.l_norms):
            self.indiv_adversary.eps = self.eps[i] + 0.
            self.indiv_adversary.step_size = self.step_size[i] + 0.
            self.indiv_adversary.norm = norm + ''
            self.indiv_adversary.n_iter = self.n_iter[i]
            if not return_acc:
                x_curr = self.indiv_adversary.perturb(model, x, y)
            else:
                x_curr, acc_curr = self.indiv_adversary.perturb(model, x, y, return_acc=True)
                l_acc.append(acc_curr)
            l_x_adv.append(x_curr.clone())
        
        if not self.use_miscl:
            if not return_acc:
                return torch.cat(l_x_adv, dim=0)
            else:
                return torch.cat(l_x_adv, dim=0), torch.cat(l_acc, dim=0)
        else:
            #logits = torch.zeros([len(l_x_adv), x.shape[0], 10]).cuda()
            loss = torch.zeros([len(l_x_adv), x.shape[0]]).cuda()
            for i, x_adv in enumerate(l_x_adv):
                output = model(x_adv)
                loss[i] = self.loss(output, y) - 1e5 * (output.max(dim=1)[1] == y).float()
            ind_max = loss.max(dim=0)[1]
            x_adv = l_x_adv[0].clone()
            x_adv[ind_max == 1] = l_x_adv[1][ind_max == 1].clone()
            x_adv[ind_max == 2] = l_x_adv[2][ind_max == 2].clone()
            
            return x_adv

# data loaders
def load_data(args):
    crop_input_size = 0 if args.crop_input is None else args.crop_input
    crop_data_size = 0 if args.crop_data is None else args.crop_data
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32 - crop_data_size, padding=4 - crop_input_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([
            #transforms.RandomCrop(32 - crop_data_size, padding=4 - crop_input_size),
            transforms.ToTensor(),
        ])
    elif args.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    root = args.data_dir + '' #'/home/EnResNet/WideResNet34-10/data/'
    num_workers = 2
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            root, train=False, transform=test_transform, download=True)
    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(root=root, split='train',
            transform=train_transform, download=True)
        test_dataset = datasets.SVHN(root=root, split='test', #train=False
            transform=test_transform, download=True)
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(
            root, train=True, transform=train_transform, download=True)
        test_dataset = datasets.MNIST(
            root, train=False, transform=test_transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    return train_loader, test_loader
    
def load_imagenet_train(args):
    from robustness.datasets import DATASETS
    from robustness.tools import helpers
    data_paths = ['/home/scratch/datasets/imagenet',
        '/scratch_local/datasets/ImageNet2012',
        '/mnt/qb/datasets/ImageNet2012',
        '/scratch/datasets/imagenet/']
    for data_path in data_paths:
        if os.path.exists(data_path):
            break
    print(f'found dataset at {data_path}')
    dataset = DATASETS['imagenet'](data_path) #'/home/scratch/datasets/imagenet'
    
    
    train_loader, val_loader = dataset.make_loaders(2,
                    args.batch_size, data_aug=True)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    return train_loader, val_loader

# other utils
def get_accuracy(model, data_loader=None):
    assert not model.training
    if not data_loader is None:
        acc = 0.
        c = 0
        with torch.no_grad():
            for (x, y) in data_loader:
                output = model(x.cuda())
                acc += (output.cpu().max(dim=1)[1] == y).float().sum()
                c += x.shape[0]
    return acc.item() / c

def get_lr_schedule(args):
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
        # lr_schedule = lambda t: np.interp([t], [0, args.epochs], [0, args.lr_max])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule.startswith('piecewise'):
        w = [float(c) for c in args.lr_schedule.split('-')[1:]]
        def lr_schedule(t):
            c = 0
            while t / args.epochs > sum(w[:c + 1]) / sum(w):
                c += 1
            return args.lr_max / 10. ** c
    return lr_schedule

def norm_schedule(it, epoch, epochs, l_norms, ps=None, schedule='piecewise'):
    #assert l_norms == ['Linf', 'L2', 'L1']
    if schedule == 'piecewise':
        if epoch < epochs * .5:
            return l_norms.index('L2')
        else:
            if not ps is None:
                ind_linf = l_norms.index('Linf')
                ind_l1 = l_norms.index('L1')
                return random.choices([ind_linf, ind_l1], weights=[
                    ps[ind_linf], ps[ind_l1]])[0]
            if it % 2 == 0:
                return l_norms.index('Linf')
            else:
                return l_norms.index('L1')


# swa tools
def polyak_averaging(p_local, p_new, it):
    return (it * p_local + p_new) / (it + 1.)

def exp_decay(p_local, p_new, theta=.995):
    return theta * p_local + (1. - theta) * p_new

class AveragedModel(nn.Module):
    def __init__(self, model):
        super(AveragedModel, self).__init__()
        self._model = copy.deepcopy(model)
        #
    
    def forward(self, x):
        return self._model(x)

    @torch.no_grad()
    def update_parameters(self, model, avg_fun=exp_decay):
        for p_local, p_new in zip(self._model.parameters(), model.parameters()):
            p_local.set_(avg_fun(p_local, p_new))


# initializations
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        n = module.in_features
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

# stepsizes for pgd-at
def get_stepsize(norm, eps, method='default'):
    #assert method == 'default'
    if method == 'default':
        if norm == 'Linf':
            return eps / 4.
        elif norm == 'L2':
            return eps / 3.
        elif norm == 'L1':
            return 2. * eps * 255. / 2000.
        else:
            raise ValueError('please specify a norm')
    elif method == 'msd':
        if norm == 'Linf':
            return eps / 4.
        elif norm == 'L2':
            return eps / 3.
        elif norm == 'L1':
            return 1. #2. * eps * 255. / 2000.
        else:
            raise ValueError('please specify a norm')
    elif method == 'msd-5':
        if norm == 'Linf':
            return eps / 2.
        elif norm == 'L2':
            return eps / 1.5
        elif norm == 'L1':
            return eps / 2.
        else:
            raise ValueError('please specify a norm')
    elif method == 'half':
        return eps / 2.

# utils max strategy
def form_batch_max(l_adv, l_acc, l_loss, l_norm):
    bs = l_adv[0].shape[0]
    adv = l_adv[0].clone()
    best_norm = torch.zeros([bs]).long() #[ for _ in range(bs)]
    best_loss = l_loss[0].clone()
    best_acc = l_acc[0].clone()
    for counter in range(1, len(l_norm)):
        ind = l_loss[counter] > best_loss
        adv[ind] = l_adv[counter][ind].clone()
        best_norm[ind] = counter + 0
        best_loss[ind] = l_loss[counter][ind].clone()
        best_acc[ind] = l_acc[counter][ind].clone()
    #best_norm = [l_norm[best_norm[val].item()] + '' for val in range(bs)]
    
    return adv, best_norm, best_acc, best_loss

def random_crop(x, size, padding):
    z = torch.zeros([x.shape[0], x.shape[1], size + 2 * padding,
        size + 2 * padding], device=x.device)
    z[:, :, padding:padding + size, padding:padding + size] += x
    
    a = random.randint(0, 2 * padding)
    b = random.randint(0, 2 * padding)

    return z[:, :, a:a + size, b:b + size]


class BatchTracker():
    def __init__(self, imgs, labs, bs, norms, alpha):
        self.imgs_orig = imgs.clone()
        self.labs_orig = labs.clone()
        self.bs = bs
        self.n_ex = imgs.shape[0]
        self.norms = norms
        #self.loss_norms = torch.zeros([self.n_ex, 2]) #{k: torch.zeros([self.n_ex]) for k in norms}
        #self.count_norms = self.loss_norms.clone()
        self.loss_norms_ra = torch.zeros([self.n_ex, 2])
        self.alpha = alpha
    
    def batch_new_epoch(self):
        self.ind_sort = torch.randperm(self.n_ex)
        self.batch_init = 0
        #self.loss_norms[k]
        u = torch.ones_like(self.loss_norms_ra[:, 0])
        '''ps = (self.loss_norms[:, 0] / torch.max(self.count_norms[:, 0], u)) / torch.max(
            self.loss_norms[:, 0] / torch.max(self.count_norms[:, 0], u
            ) + self.loss_norms[:, 1] / torch.max(self.count_norms[:, 1], u), u)
        ps = torch.max(ps, .1 * u)'''
        
        tot_curr = self.loss_norms_ra[:, 0] + self.loss_norms_ra[:, 1]
        ind_tot_curr = tot_curr == 0.
        tot_curr[tot_curr == 0.] = 1.
        ps = self.loss_norms_ra[:, 0] / tot_curr
        ps_old = ps.clone()
        #ps = ps * (self.loss_norms_ra.min(dim=1)[0] > 0.) + .5 * (self.loss_norms_ra.min(dim=1)[0] <= 0.)
        ps[(ps == 0.) + (ps == 1.)] = 1. - ps[(ps == 0.) + (ps == 1.)]
        if True: #False
            #
            ps = (self.loss_norms_ra[:, 0] > self.loss_norms_ra[:, 1]).float() #ps.min(2)[1]
            ps[ps_old == 0.] = 1.
            ps[ps_old == 1.] = 0.
        
        #print(ps)
        ps[ind_tot_curr] = .5
        
        #print(ps)
        #print(ind_tot_curr.sum(), ((ps_old == 0.) + (ps_old == 1.)).sum())
        #norm_at = (ps < random.random()).long()
        #print(self.labs_orig)
        #print(norm_at)
        
        train_loader = []
        for c in range(0, self.n_ex, self.bs):
            ind_curr = self.ind_sort[c:c + self.bs].clone()
            x_curr = self.custom_augm(self.imgs_orig[ind_curr].clone())
            y_curr = self.labs_orig[ind_curr].clone()
            norm_curr = (ps[ind_curr] < random.random()).long().clone() #norm_at[ind_curr].clone()
            #print(y_curr, norm_curr)
            train_loader.append((x_curr.clone(), y_curr, norm_curr))
        return train_loader

    def custom_augm(self, x):
        z = random_crop(x, x.shape[-1], 4)
        if random.random() > .5:
            return transforms.functional.hflip(z)
        else:
            return z.clone()
    
    def update_loss(self, loss, norm, i):
        ind_curr = self.ind_sort[i * self.bs:(i + 1) * self.bs].clone()
        #print(ind_curr)
        #self.loss_norms[ind_curr, norm] += loss
        #self.count_norms[ind_curr, norm] += 1
        self.loss_norms_ra[ind_curr, norm] = self.loss_norms_ra[ind_curr, norm
            ] * self.alpha + loss.cpu() * (1. - self.alpha)


# different resolution
class ImageCropper(nn.Module):
    def __init__(self, w: int, #float, float, float
        shape: Tuple[int, int, int, int]) -> None:
        super(ImageCropper, self).__init__()

        mask = get_mask(shape, w)
        self.register_buffer('mask', mask)

    def forward(self, input: Tensor) -> Tensor:
        return input * self.mask


def get_mask(shape, w):
    mask = torch.zeros(shape)
    assert len(mask.shape) == 4
    mask[:, :, w:mask.shape[-2] - w, w:mask.shape[-1] - w] = 1

    return mask


def add_preprocessing(model: nn.Module, w: int, shape: Tuple[int, int, int]
    ) -> nn.Module:
    layers = OrderedDict([
        ('crop', ImageCropper(w, [1] + shape)),
        ('model', model)
    ])
    return nn.Sequential(layers)


def Lp_norm(x, p, keepdim=False):
    assert p > 1
    z = x.view(x.shape[0], -1)
    t = (z.abs() ** p).sum(dim=-1) ** (1. / p)
    if keepdim:
        t = t.view(-1, *[1] * (len(x.shape) - 1))
    return t


if __name__ == '__main__':
    '''args = lambda: 0
    args.batch_size = 256
    train_loader, test_loader = load_imagenet_train(args)
    print(len(train_loader), len(test_loader))'''

    custom_loader = BatchTracker(torch.ones([10, 3, 5, 5]) * torch.arange(
        10).view(-1, 1, 1, 1), torch.arange(10), 3, ['Linf', 'L1'], .9)
    for epoch in range(4):
        startt = time.time()
        train_loader = custom_loader.batch_new_epoch()
        print('loader created in {:.3f} s'.format(time.time() - startt))
        loss = torch.zeros([10])
        #norm_all = loss.clone()
        #print(custom_loader.loss_norms_ra)
        for i in range(4): #(x, y, norm) in enumerate(train_loader)
            x, y, norm = train_loader[i]
            #print(y, norm)
            loss = torch.randn([y.shape[0]]).abs()
            custom_loader.update_loss(loss, norm, i)
            print(x.view(x.shape[0], -1).max(dim=1)[0])
        #print(custom_loader.loss_norms)
        #print(custom_loader.count_norms)
        #print(custom_loader.loss_norms_ra)

