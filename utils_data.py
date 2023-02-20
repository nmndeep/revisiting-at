from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.transforms import *
from ffcv.fields.basics import IntDecoder

import torchvision as tv
import torch as ch
from typing import List
from pathlib import Path
import numpy as np
import robustbench as rb
import autoattack

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
    
    return str_det

def early_stop(model, x, y, norm='Linf', eps=4./255., bs=1000,
    log_path=None, sd=0, verbose=True, iters=1):
    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
        log_path=None, seed=sd)
    adversary.attacks_to_run = ['apgd-ce']
    adversary.apgd.n_restarts=1
    adversary.apgd.n_iter=iters
    with ch.no_grad():
        x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    #if verbose
    acc = rb.utils.clean_accuracy(model, x_adv, y, device='cuda')
    # check_imgs(x_adv, x, norm)
    print('robust accuracy: {:.1%}'.format(acc))
    return acc


def get_loaders(n_ex, args, only_val=False, on_cpu=False, shuffle=True,
    use_channel_last=True):
    '''train_loader = Loader(args.data_dir, batch_size=512, 
    num_workers=8, order=OrderOption.RANDOM,
    pipelines={'image': [
            RandomResizedCropRGBImageDecoder((224, 224)),
            ToTensor(), 
            # Move to GPU asynchronously as uint8:
            ToDevice(ch.device('cuda:0'), non_blocking=True), 
            # Automatically channels-last:
            ToTorchImage(), 
            Convert(ch.float16), 
            # Standard torchvision transforms still work!
            #tv.transforms.Normalize(MEAN, STDEV)
        ]})'''
    if not only_val:
        train_loader = create_train_loader(gpu=0,
            train_dataset=f'{args.data_dir}/train_400_0.50_90.ffcv',
            num_workers=args.num_workers, batch_size=args.batch_size,
            distributed=args.distributed, in_memory=True, res=args.img_size,
            mean=(0., 0., 0.), std=(1., 1., 1.), prec=args.precision)
    else:
        train_loader = None
        
    val_loader = create_val_loader(gpu=0,
        val_dataset=f'{args.data_dir}/val_400_0.50_90.ffcv',
        num_workers=args.num_workers, batch_size=args.batch_size,
        distributed=args.distributed, resolution=args.img_size,
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_ratio=224 / 256,
        prec=args.precision, on_cpu=on_cpu, shuffle=shuffle,
        use_channel_last=use_channel_last)
        
    return train_loader, val_loader


def create_train_loader(gpu, train_dataset, num_workers, batch_size,
    distributed, in_memory, res, mean, std, prec):
    this_device = f'cuda:{gpu}'
    train_path = Path(train_dataset)
    assert train_path.is_file()
    IMAGENET_MEAN, IMAGENET_STD = np.array(mean) * 255., np.array(std) * 255.

    #res = self.get_resolution(epoch=0)
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        #RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(ch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, prec)
    ]
    
    #np.random.seed(0)

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), non_blocking=True)
    ]

    #np.random.seed(0)
    
    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed
                    )

    return loader
    

def create_val_loader(gpu, val_dataset, num_workers, batch_size,
    resolution, distributed, mean, std, crop_ratio, prec, on_cpu=False,
    shuffle=False, use_channel_last=True):
    this_device = f'cuda:{gpu}' if not on_cpu else 'cpu'
    val_path = Path(val_dataset)
    assert val_path.is_file()
    IMAGENET_MEAN, IMAGENET_STD = np.array(mean) * 255., np.array(std) * 255.
    DEFAULT_CROP_RATIO = crop_ratio
    print('using mean and std:', mean, std)
    
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    if use_channel_last:
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, prec)
        ]
    else:
        print('not using channel last memory format')
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(channels_last=False),
            Convert(ch.cuda.FloatTensor),
            tv.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device),
        non_blocking=True)
    ]

    order = OrderOption.SEQUENTIAL if not shuffle else OrderOption.RANDOM
    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order, #OrderOption.SEQUENTIAL
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed,
                    seed=0
                    )
    return loader
