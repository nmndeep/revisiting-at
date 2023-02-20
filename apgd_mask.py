import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple
from functools import partial
from random import shuffle

try:
    from other_utils import L1_norm, L2_norm, L0_norm, Logger
except ImportError:
    from autoattack.other_utils import L1_norm, L2_norm, L0_norm, Logger


def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def cw_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind))


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


criterion_dict = {'ce': lambda x, y: F.cross_entropy(x, y, reduction='none'),
    'dlr': dlr_loss,
    'cw': cw_loss,
    'dlr-targeted': dlr_loss_targeted,
    'l2': lambda x, y: -1. * L2_norm(x - y) ** 2.,
    'l1': lambda x, y: -1. * L1_norm(x - y),
    'linf': lambda x, y: -1. * (x - y).abs().max(-1)[0],
    }


def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
      t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def get_target_features(model, x, y):
    """ for an input x returns the features of a point z which is classified
        in the second most likely class for x if x correctly classified, in
        the first one otherwise
    """
    logits, fts = model.forward(x, return_fts=True)
    pred = logits.sort(dim=-1)[1]
    fts_target = fts.clone()
    ind = [c for c in range(x.shape[0])]
    shuffle(ind)
    for c in range(x.shape[0]):
        if pred[c][-1] != y[c]:
            continue
        fts_secondclass = False
        for i in ind:
            if pred[i][-1] == pred[c][-2]:
                fts_target[c] = fts[i].clone()
                fts_secondclass = True
                break
        # make sure that fts from another class are used
        if not fts_secondclass:
            for i in ind:
                if pred[i][-1] != y[c]:
                    fts_target[c] = fts[i].clone()
                    break
    return fts_target

    
def get_mask(mask_type='patch', sh_patch=[16, 16], pos=[[0, 0]], sh_image=[3, 224, 224],
    device='cuda', offset=[0, 0], margin=[0, 0], n_copies=1, width=1):
    """ patches
        sh_patches: size of cells
        offset: the position of the corner of the top left cell
        margin: the position of the patches inside the cell (patches are centered)
        
        frames
        width: size of frames
    """
    if mask_type == 'allonsingleimage':
        #n_patches = int((sh_image[1] / sh_patch[0] * sh_image[2] / sh_patch[1])
        n_hor = math.floor((sh_image[1] - offset[0]) / sh_patch[0])
        n_ver = math.floor((sh_image[2] - offset[1]) / sh_patch[1])
        n_patches = n_hor * n_ver
        m = torch.zeros([n_patches, *sh_image], device=device)
        i_patch = 0
        '''for c in range(math.ceil(sh_image[1] / sh_patch[0])):
            for i in range(math.ceil(sh_image[2] / sh_patch[1])):
                m[i_patch, :, c * sh_patch[0]:(c + 1) * sh_patch[0], 
                    i * sh_patch[1]:(i + 1) * sh_patch[1]] = 1.
                i_patch += 1
        '''
        for c in range(n_hor):
            for i in range(n_ver):
                m[i_patch, :, offset[0] + c * sh_patch[0] + margin[0]:offset[0] + (c + 1) * sh_patch[0] - margin[0], 
                    offset[1] + i * sh_patch[1] + margin[1]:offset[1] + (i + 1) * sh_patch[1] - margin[1]] = 1.
                i_patch += 1
        assert i_patch == n_patches
        
        return m
        
    elif mask_type == 'allongrid':
        sh = [sh_patch[0] - margin[0], sh_patch[1] - margin[1]]
        print(f'using patches {sh[0]}x{sh[1]} on grid {sh_patch[0]}x{sh_patch[1]}')
        n_hor = math.floor((sh_image[1] - offset[0]) / sh[0])
        n_ver = math.floor((sh_image[2] - offset[1]) / sh[1])
        grid_size = [math.ceil(sh_image[c + 1] / sh_patch[c]) - 1 for c in range(2)]
        n_patches = n_hor * grid_size[1] + n_ver * grid_size[0]
        m = torch.zeros([n_patches, *sh_image], device=device)
        i_patch = 0
        for c in range(n_hor):
            for e in range(1, grid_size[1] + 1):
                m[i_patch, :, offset[0] + c * sh[0]:offset[0] + (c + 1) * sh[0],
                    sh_patch[1] * e - sh[1] // 2:sh_patch[1] * e + sh[1] // 2] = 1.
                i_patch += 1
        for c in range(n_ver):
            for e in range(1, grid_size[0] + 1):
                m[i_patch, :, sh_patch[0] * e - sh[0] // 2:sh_patch[0] * e + sh[0] // 2,
                    offset[1] + c * sh[1]:offset[1] + (c + 1) * sh[1]] = 1.
                i_patch += 1
        assert i_patch == n_patches
        #print(f'initially {n_patches} positions')
        l_uniques = [0]
        for c in range(1, n_patches):
            t = (m[:c, 0] - m[c, 0].unsqueeze(0)).abs().view(c, -1).sum(-1)
            #print(t)
            if t.min() > 0:
              l_uniques.append(c)
        m = m[l_uniques]
        print(f'{m.shape[0]} positions (initially {n_patches})')
        
        return m
    
    elif mask_type == 'posonimage':
        n_patches = len(pos) * n_copies
        print(f'using patches {sh_patch[0]}x{sh_patch[1]}')
        m = torch.zeros([n_patches, *sh_image], device=device)
        for i_patch, c in enumerate(pos):
            print(f'pos {i_patch} at ({c[0]}, {c[1]})')
            for i_copy in range(n_copies):
                m[i_patch * n_copies + i_copy, :, c[0]:c[0] + sh_patch[0],
                    c[1]:c[1] + sh_patch[1]] = 1.
        
        return m
        
    elif mask_type == 'frame':
        m = torch.zeros([1, *sh_image], device=device)
        m[:, :, :width] = 1.
        m[:, :, -width:] = 1.
        m[:, :, :, :width] = 1.
        m[:, :, :, -width:] = 1.
        print(f'{m[0, 0].sum():.0f} pixels in frame')
        
        return m
    
    else:
        raise ValueError()
        

def shift_mask(mask, shift=[0, 0], keepsize=False):
    """ shifts all masks in the batch of shift
        (individual shifts not implemented)
        if keepsize, the mask can't go over image borders, assuming
        rectangular maskes constant over channels
    """
    sh = mask.shape
    assert len(sh) == 4
    m = torch.zeros([sh[0], sh[1], sh[2] + 2 * abs(shift[0]), sh[3] + 2 * abs(shift[1])],
            device=mask.device)
    m[:, :, abs(shift[0]):abs(shift[0]) + sh[2], abs(shift[1]):abs(shift[1]
        ) + sh[3]] = mask.clone()
    sp = [max(0, -2 * c) for c in shift]
    #print(sp)
    shifted_mask = m[:, :, sp[0]:sp[0] + sh[2], sp[1]:sp[1] + sh[3]].clone()
    if not keepsize:
        return shifted_mask
    for i, im in enumerate(shifted_mask):
        #print(im)
        im = im > 0
        if im.sum() == (mask[i] > 0).sum():
            continue
        # size of the mask
        h = (mask[i, 0] > 0).sum(dim=0).max()
        v = (mask[i, 0] > 0).sum(dim=1).max()
        h_cropped = h - im[0].sum(dim=0).max()
        v_cropped = v - im[0].sum(dim=1).max()
        #print(h_cropped, v_cropped)
        sp[0] = max(0, -2 * shift[0]) + math.copysign(h_cropped, shift[0])
        sp[1] = max(0, -2 * shift[1]) + math.copysign(v_cropped, shift[1])
        sp = [int(c) for c in sp]
        #print(sp)
        shifted_mask[i] = m[i, :, sp[0]:sp[0] + sh[2], sp[1]:sp[1] + sh[3]]
    return shifted_mask

    
def shift_patch_on_image(im, mask, orig_im, shift, keepsize=True):
    """ shifts all patches in the batch over the original images of shift
    """
    shifted_mask = shift_mask(mask, shift, keepsize)
    shifted_im = shift_mask(mask * im, shift, keepsize)
    return orig_im * (1. - shifted_mask) + shifted_im, shifted_mask




def apgd_train(model, x, y, norm='Linf', eps=8. / 255., n_iter=100,
    use_rs=False, loss='ce', verbose=False, is_train=False, logger=None,
    early_stop=None, y_target=None, fts_target=None, mask=None, alpha=None,
    optimize_mask=None, update_mask_iters=[], shifts=[], x_init=None,
    show_maxloss=False):
    assert not model.training
    device = x.device
    ndims = len(x.shape) - 1
    n_cls = 10
    
    if logger is None:
        logger = Logger()
    loss_name = loss + ''
    if mask is None:
        mask = torch.ones_like(x, requires_grad=False)
        assert not optimize_mask
    else:
        logger.log('using mask')
        if not optimize_mask is None:
            logger.log(f'optimizing location of mask {optimize_mask}')
    logger.log(f'using early stopping={early_stop}')
    
    # initialization
    if not use_rs:
        x_adv = x.clone()
    else:
        #raise NotImplemented
        if norm == 'Linf':
            t = (torch.rand_like(x) - .5) * 2. * eps
            t *= mask # add noise only on mask
            x_adv = (x + t).clamp(0., 1.)
        elif norm == 'L2':
            t = torch.randn_like(x)
            t *= mask # noise only on mask
            x_adv = x + eps * t / (L2_norm(t, keepdim=True) + 1e-12)
            x_adv.clamp_(0., 1.)
    if not x_init is None:
        x_adv = x_init.clone()
        assert x_init.shape == x.shape, 'init and input shapes not matching'
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    mask_best = mask.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    loss_adv = -float('inf') * torch.ones(x.shape[0], device=device)
    
    # set loss
    if not loss in ['dlr-targeted']:
        criterion_indiv = criterion_dict[loss]
    else:
        assert not y_target is None
        criterion_indiv = partial(criterion_dict[loss], y_target=y_target)
    
    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2. if alpha is None else alpha
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old =  n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1. if alpha is None else alpha
    
    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
        device=device)
    counter3 = 0

    x_adv.requires_grad_()
    if not loss in ['l2', 'l1', 'linf']:
        logits = model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        loss = loss_indiv.sum()
    else:
        logits, fts = model.forward(x_adv, return_fts=True)
        loss_indiv = criterion_indiv(fts, fts_target)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    accum_grad = grad.clone()
    grad *= mask
    
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0
    
    u = torch.arange(x.shape[0], device=device)
    e = torch.ones_like(x)
    x_adv_old = x_adv.clone().detach()
    skip_momentum = False
    
    for i in range(n_iter):
        ### gradient step
        if True: #with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach() #.mean()
            
            a = 0.75 if i > 0 else 1.0
            #a = .5 if i > 0 else 1.
            a = 1. if skip_momentum else a
            
            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * grad / (L2_norm(grad,
                    keepdim=True) + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                    -1, 1, 1, 1) + 1e-10)
                
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p
                
            elif norm == 'L0':
                L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
                    dim=-1, keepdim=True) + 1e-12).view(grad.shape[0], *[1]*(
                    len(grad.shape) - 1))
                x_adv_1 = x_adv + step_size * L1normgrad * n_fts
                x_adv_1 = L0_projection(x_adv_1, x, eps)
                # TODO: add momentum
                
            
            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        if not loss_name in ['l2', 'l1', 'linf']:
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
        else:
            logits, fts = model.forward(x_adv, return_fts=True)
            loss_indiv = criterion_indiv(fts, fts_target)
            loss = loss_indiv.sum()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            accum_grad = (accum_grad * (i + 1) + grad) / (i + 2)
            grad *= mask
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        
        # collect points and stats
        pred = logits.detach().max(1)[1] == y
        acc_old = acc.clone()
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0) * (acc_old == 1.) + (~pred) * (
            loss_indiv.detach().clone() > loss_adv) #.nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        loss_adv[ind_pred] = loss_indiv.detach().clone()[ind_pred]
        
        # logging
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean())
            str_stats += f' - max L0 pert: {L0_norm(x - x_adv).max():.0f}'
            if show_maxloss:
                str_stats += f' - max loss {loss_best.max():.6f}'
            logger.log('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1 + 0
          ind = (y1 > loss_best).nonzero().squeeze()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0
          mask_best[ind] = mask[ind].clone()
          
          if i in update_mask_iters: #and i < n_iter / 2
              # shift patch on image and mask on gradients
              
              # TODO: the gradient from old location is used, might be better to
              # use the new one, but would require an extra bp
            
              shift = next(shifts)
              signed_shift = [random.choice([-1, 0, 1]) * shift for _ in range(2)]
              
              if optimize_mask == 'shift_patch':
                  # move the patch on the image
                  x_adv, mask_curr = shift_patch_on_image(x_adv, mask, x,
                      shift=signed_shift, keepsize=True)
                  grad = shift_mask(grad, signed_shift, True)
                  x_adv_old, _ = shift_patch_on_image(x_adv_old, mask, x,
                      shift=signed_shift, keepsize=True)
              
              elif optimize_mask == 'shift_window':
                  # move the mask but not the patch (new area estimated with
                  # accumulated gradient
                  mask_curr = shift_mask(mask, signed_shift, True)
                  x_adv = x_adv * mask * mask_curr + accum_grad.sign(
                      ).clamp(0., 1.) * (1. - mask) * mask_curr + x * (1. - mask_curr)
                  x_adv_old = x_adv_old * mask * mask_curr + accum_grad.sign(
                      ).clamp(0., 1.) * (1. - mask) * mask_curr + x * (1. - mask_curr)
                  grad = grad * mask_curr + accum_grad * (1. - mask) * mask_curr
                      
              else:
                  raise ValueError(f'{optimize_mask} non available')
                  
              mask = mask_curr.clone()
              '''print(f'mask {L0_norm(mask).max():.0f}')
              print(f'grad {L0_norm(grad).max():.0f}')
              print(f'curr {L0_norm(x - x_adv).max():.0f}')
              print(f'prev {L0_norm(x - x_adv_old).max():.0f}')
              print(signed_shift)'''
              
          skip_momentum = False

          counter3 += 1

          if counter3 == k:
              if norm in ['Linf', 'L2']:
                  fl_oscillation = check_oscillation(loss_steps, i, k,
                      loss_best, k3=thr_decr)
                  fl_reduce_no_impr = (1. - reduced_last_check) * (
                      loss_best_last_check >= loss_best).float()
                  fl_oscillation = torch.max(fl_oscillation,
                      fl_reduce_no_impr)
                  reduced_last_check = fl_oscillation.clone()
                  loss_best_last_check = loss_best.clone()

                  if fl_oscillation.sum() > 0:
                      ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                      step_size[ind_fl_osc] /= 2.0
                      n_reduced = fl_oscillation.sum()

                      x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                      grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                      mask[ind_fl_osc] = mask_best[ind_fl_osc].clone()
                  
                  counter3 = 0
                  k = max(k - size_decr, n_iter_min)
              
              elif norm == 'L1':
                  # adjust sparsity
                  sp_curr = L0_norm(x_best - x)
                  fl_redtopk = (sp_curr / sp_old) < .95
                  topk = sp_curr / n_fts / 1.5
                  step_size[fl_redtopk] = alpha * eps
                  step_size[~fl_redtopk] /= adasp_redstep
                  step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                  sp_old = sp_curr.clone()
              
                  x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                  grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  mask[fl_redtopk] = mask_best[fl_redtopk].clone()
              
                  counter3 = 0
                  
              if not optimize_mask is None:
                  # moving the mask doesn't get along with momentum when
                  # restarting from best loss point
                  skip_momentum = True

        if (acc.sum() == 0. and early_stop == 'all') or (
            (acc == 0).any() and early_stop == 'any'):
            break
            
        
    
    return x_best, acc, loss_best, x_best_adv


def apgd_restarts(model, x, y, norm='Linf', eps=8. / 255., n_iter=10,
    loss='ce', verbose=False, n_restarts=1, log_path=None, early_stop=None,
    mask=None, bs=1000, alpha=None, optimize_patch=None, op_iters=10,
    op_update_t=5, op_topk=.1, show_maxloss=False):
    """ run apgd with the option of restarts
    """
    acc = torch.ones([x.shape[0]], dtype=bool, device=x.device) # run on all points
    x_adv = x.clone()
    x_best = x.clone()
    loss_best = -float('inf') * torch.ones_like(acc).float()
    y_target = None
    fts_target = None
    if loss in ['dlr-targeted']:
        with torch.no_grad():
            output = model(x)
        outputsorted = output.sort(-1)[1]
        n_target_classes = 4 # max number of target classes to use
    elif loss in ['l2', 'l1', 'linf']:
        with torch.no_grad():
            fts_target = get_target_features(model, x, y)
    if not mask is None:
        if isinstance(mask, str):
            mask = get_mask(mask)
        assert mask.shape == x.shape, 'mask and input shapes not matching'
    logger = Logger(log_path)
    logger.log(f'using {n_iter} iters and {n_restarts} restarts')
    
    for i in range(n_restarts):
        if acc.sum() > 0:
            if loss in ['dlr-targeted']:
                y_target = outputsorted[:, -(i % n_target_classes + 2)]
                y_target = y_target[acc]
                print(f'target class {i % n_target_classes + 2}')
            elif loss in ['l2', 'l1', 'linf']:
                with torch.no_grad():
                    fts_target = get_target_features(model, x[acc], y[acc])
            
            n_batches = math.ceil(acc.sum() / bs)
            x_adv_curr, x_best_curr = x[acc].clone(), x[acc].clone()
            loss_curr = torch.zeros_like(y[acc]).float()
            for c in range(n_batches):
                rng = range(c * bs, min((c + 1) * bs, acc.sum().item()))
                ind = torch.nonzero(acc).squeeze(1)[rng]
                print(f'batch {c}, rng {rng}, ind {ind}')
                x_best_curr[rng], acc_curr, loss_curr[rng], x_adv_curr[rng] = apgd_train(
                    model, x[ind], y[ind],
                    n_iter=n_iter, use_rs=True, verbose=verbose, loss=loss,
                    eps=eps, norm=norm, logger=logger, early_stop=early_stop,
                    y_target=y_target[ind] if not y_target is None else None,
                    fts_target=fts_target, mask=mask[ind],
                    alpha=alpha)
                if (acc_curr == 0).any() and early_stop == 'any':
                    break
                    
            if not optimize_patch is None and not ((acc_curr == 0).any() and early_stop == 'any'):
                topk = int(op_topk) if op_topk > 1. else int(loss_curr.shape[0] * op_topk)
                print(f'optimizing location for {topk} patches')
                op_ind = loss_curr.sort()[1][-topk:]
                print(op_ind)
                x_init = x_best_curr[op_ind].clone()
                op_mask = mask[op_ind].clone()
                times = list(range(op_update_t - 1, op_iters + op_update_t, op_update_t))
                shifts = [1 for _ in times]
                ind = torch.nonzero(acc).squeeze(1)
                x_best_curr[op_ind], acc_curr, loss_curr[op_ind], x_adv_curr[op_ind] = apgd_train(
                    model, x[ind][op_ind], y[ind][op_ind],
                    n_iter=op_iters, use_rs=True, verbose=verbose, loss=loss,
                    eps=eps, norm=norm, logger=logger, early_stop=early_stop,
                    y_target=y_target[ind][op_ind] if not y_target is None else None,
                    fts_target=fts_target, mask=op_mask,
                    alpha=alpha, optimize_mask=optimize_patch, update_mask_iters=times,
                    shifts=iter(shifts), x_init=None, show_maxloss=show_maxloss)
                
            
            with torch.no_grad():
                acc_curr = model(x_adv_curr).max(1)[1] == y[acc]
            succs = torch.nonzero(acc).squeeze()
            if len(succs.shape) == 0:
                succs.unsqueeze_(0)
            x_adv[succs[~acc_curr]] = x_adv_curr[~acc_curr].clone()
            # old version
            '''ind = succs[acc_curr * (loss_curr > loss_best[acc])]
            x_best[ind] = x_best_curr[acc_curr * (loss_curr > loss_best[acc])].clone()
            loss_best[ind] = loss_curr[acc_curr * (loss_curr > loss_best[acc])].clone()'''
            # new version
            ind = succs[loss_curr > loss_best[acc]]
            x_best[ind] = x_best_curr[loss_curr > loss_best[acc]].clone()
            loss_best[ind] = loss_curr[loss_curr > loss_best[acc]].clone()
            
            ind_new = torch.nonzero(acc).squeeze()
            acc[ind_new] = acc_curr.clone()
            
            print(f'restart {i + 1} robust accuracy={acc.float().mean():.1%}')
            
            if (acc == 0).any() and early_stop == 'any':
                break
            
    # old version
    #x_best[~acc] = x_adv[~acc].clone()
    
    
    return x_adv, loss_best, x_best



    
    