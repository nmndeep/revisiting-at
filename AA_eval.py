import torch
from torchvision import models
import torch.nn as nn

from timm.models import create_model
from torchvision import datasets, transforms
import math
import argparse
import os
import sys
sys.path.insert(0,'..')

import json
import robustbench
import numpy as np

from autoattack import AutoAttack
from robustbench.utils import clean_accuracy

from main import BlurPoolConv2d, PREC_DICT, IMAGENET_MEAN, \
    IMAGENET_STD
from utils_data import get_loaders, early_stop
from utils_architecture import normalize_model, get_new_model, interpolate_pos_encoding
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

def sizeof_fmt(num, suffix="Flops"):
    for unit in ["", "Ki", "Mi", "G", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.3f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"

eps_dict = {'imagenet': {'Linf': 4. / 255., 'L2': 2., 'L1': 75.}}


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log, verbose=False):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log)
                f.write('\n')
                if verbose:
                    f.flush()

def format(value):
    return "%.3f" % value


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--model', default='convnext_tiny_convmlp_nolayerscale', type=str)
    parser.add_argument('--n_ex', type=int, default=5000)
    parser.add_argument('--norm', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--data_dir', type=str, default='/scratch/fcroce42/ffcv_imagenet_data')
    parser.add_argument('--only_clean', action='store_true')
    parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--precision', type=str, default='fp32')
    parser.add_argument('--ckpt_path', type=str, default='/scratch/nsingh/ImageNet_Arch/model_2022-12-04 14:36:56_convnext_iso_iso_0_not_orig_0_pre_1_aug_0_adv__50_at_crop_flip/weights_18.pt')
    parser.add_argument('--mod', type=str)
    parser.add_argument('--model_in', nargs='+')
    parser.add_argument('--full_aa', type=int, default=0)
    parser.add_argument('--init', type=str)
    parser.add_argument('--add_normalization', action='store_true', default=False)
    parser.add_argument('--l_norms', type=str)
    parser.add_argument('--l_epss', type=str)
    parser.add_argument('--get_stats', action='store_true')
    parser.add_argument('--use_fixed_val_set', action='store_true', default=False)
    parser.add_argument('--img_size', type=int, help='resolution to test the evaluataion for, default: 224', default=224)
    parser.add_argument('--not_channel_last', action='store_false')
    parser.add_argument('--not-original', type=int, default=1)
    parser.add_argument('--updated', action='store_true', help='Patched models?', default=False)

    args = parser.parse_args()
    return args

def main():
    args = get_args_parser()
    
    mods = [args.mod]
    nots = [bool(args.not_original)]
    args.model_in = ' '.join(args.model_in)
    ll = [args.model_in]
    args.ckpt_path = args.model_in
    data_path = 'patch_to_imagenet_validataion_set'

    device = 'cuda'

    assert len(mods) == len(nots) == len(ll)

    print('using fixed val set')


    crop_pct = 0.875
    
    img_size = args.img_size

    scale_size = int(math.floor(img_size / crop_pct))
    trans = transforms.Compose([
        transforms.Resize(
            scale_size,
            interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    x_test_val, y_test_val = robustbench.data.load_imagenet(5000, data_dir=data_path,
        transforms_test = trans)



    print(f"{args.mod} has resolution : {img_size}")

    for idx, modd in enumerate(ll):

        args.ckpt_path += "/weights_20.pt"
        args.model = mods[idx]
        args.not_original = nots[idx]

        if not args.ckpt_path is None:
            # assert os.path.exists(args.ckpt_path), f'{args.ckpt_path} not found'
            args.savedir = '/'.join(args.ckpt_path.split('/')[:-1])
            print(args.savedir)
            # ep = args.ckpt_path.split('/')[-1].split('.pt')[0]
            with open(f'{args.savedir}/params.json', 'r') as f:
                params = json.load(f)
            args.use_blurpool = params['training.use_blurpool'] == 1
            if 'model.add_normalization' in params.keys():
                args.add_normalization = params['model.add_normalization'] == 1
            args.model = args.model #params['model.arch']
        else:
            args.savedir = './results/'
            makedir(args.savedir)
            
        args.n_cls = 1000
        args.num_workers = 1
        
        if not args.eps is None and args.eps > 1 and args.norm == 'Linf':
            args.eps /= 255.
        

        device = 'cuda'
        arch = args.model
        pretrained = False
        add_normalization = args.add_normalization

        log_path = f'{args.savedir}/evaluated_logs_{args.l_norms}_{args.full_aa}_8_255.txt'
        logger = Logger(log_path)

        print(f"Creating model: {args.model}")
        if not arch.startswith('timm_'):
                model = get_new_model(arch, pretrained=False, not_original=args.not_original, updated=args.updated)
        else:
            try:
                model = create_model(arch.replace('timm_', ''), pretrained=pretrained)
            except:
                model = get_new_model(arch.replace('timm_', ''))
        
        if add_normalization:
            print('add normalization layer')
            model = normalize_model(model, IMAGENET_MEAN, IMAGENET_STD)
       
        inpp = torch.rand(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, inpp)
        val = flops.total()
        print(sizeof_fmt(int(val)))
        print(flop_count_table(flops, max_depth=2))
        print(flops.by_operator())

        accs = []
        best_test_rob = 0.

        for i in rann:
            
            ckpt = torch.load(args.savedir + f"/model_file_name.pt", map_location='cpu') #['model']
            # print(ckpt.keys())
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
            ckpt = {k.replace('se_', 'se_module.'): v for k, v in ckpt.items()}
            model.load_state_dict(ckpt)
            model = model.to(device)
            model.eval()
            acc = clean_accuracy(model, x_test_val, y_test_val, batch_size=args.batch_size,
                    device=device)
            print(f"clean {i} : {acc}")    
        img_sizer = [img_size, img_size]
        if not args.ckpt_path is None:
            ## change the size of positional embedding in ViT if img_size>224 (training resolution).
            if "vit" in arch:
                old_shape = ckpt['pos_embed'].shape
                ckpt['pos_embed'] = interpolate_pos_encoding(
                    ckpt['pos_embed'], new_img_size=img_sizer[0],
                    patch_size=model.patch_embed.patch_size[0])
                new_shape = ckpt['pos_embed'].shape
                print(old_shape, new_shape)
                model.pos_embed = nn.Parameter(torch.zeros(new_shape, device=model.pos_embed.device))

                model.patch_embed.img_size = img_sizer
                model.patch_embed.num_patches = new_shape[1] - 1
                model.patch_embed.grid_size = (
                    img_sizer[0] // model.patch_embed.patch_size[0],
                    img_sizer[1] // model.patch_embed.patch_size[1])
        model.eval()

        str_to_log = ''
       
        logger = Logger(log_path)
        logger.log(str_to_log)

        all_norms =  [args.l_norms] #
        #all_norms = ['L2', 'L1', 'Linf']
        l_epss = [eps_dict['imagenet'][c] for c in all_norms]
        logger.log(all_norms, l_epss)
        all_acs = []
        for idx, nrm in enumerate(all_norms):
            epss = l_epss[idx]
            adversary = AutoAttack(model, norm=nrm, eps=epss,
                version='standard', log_path=log_path)
            str_to_log = ''

            if not bool(args.full_aa):
                adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

            str_to_log += f'norm={nrm} eps={l_epss[idx]:.5f}\n'
            
            assert not model.training
            
            with torch.no_grad():
                x_adv = adversary.run_standard_evaluation(x_test_val,
                        y_test_val, bs=args.batch_size)

            acc = clean_accuracy(model, x_adv, y_test_val, batch_size=args.batch_size,
                device=device)
            print('robust accuracy: {:.2%}'.format(acc))
            str_to_log += 'robust accuracy: {:.2%}\n'.format(acc)
            logger.log(str_to_log)
            all_acs.append(acc) 
            
        if args.save_imgs:
            valset = '_oldset' if args.use_fixed_val_set else ''
            runname = f'aa_short_1_{args.n_ex}_{args.norm}_{args.eps:.5f}{valset}.pth'
            savepath = f'{args.savedir}/{runname}'
            torch.save(x_adv.cpu(), savepath)      

if __name__ == '__main__':
    main()
