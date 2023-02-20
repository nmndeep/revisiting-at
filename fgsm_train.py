import torch
import torch.nn as nn
import torch.nn.functional as F

from autopgd_train_clean import criterion_dict
import robustbench as rb
#from autopgd_train import apgd_train
# import utils
# from model_zoo.fast_models import PreActResNet18
# import other_utils
import autoattack
criterion_dict = {'ce': lambda x, y: F.cross_entropy(x, y, reduction='none')}




def fgsm_attack(model, images, labels, eps) :
    
    loss = nn.CrossEntropyLoss()
    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.sum().backward()
    attack_images = images.clone()
    attack_images += eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images




def fgsm_attack(model, x, y, eps=4./255.):

    # assert not model.training

    # Set requires_grad attribute of tensor. Important for Attack
    x.requires_grad = True

    # Forward pass the data through the model
    output = model(x)
    # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    criterion_indiv = criterion_dict['ce']

    # Calculate the loss
    loss = criterion_indiv(output, y)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.sum().backward()
    # Collect datagrad
    data_grad = x.grad.data

    # Call FGSM Attack
    x_adv = gen_pert(x, eps, data_grad)

    return x_adv
    # output = model(x_adv)
    # final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    # if final_pred.item() == target.item():
    #         correct += 1
    # return correct



def fgsm_train(model, x, y, eps, loss='ce', alpha=1.25, use_rs=False,
    noise_level=1., skip_projection=False):
    assert not model.training
    
    if not use_rs:
        x_adv = x.clone()
    else:
        #raise NotImplemented
        #if norm == 'Linf'
        t = torch.rand_like(x)
        x_adv = x + (2. * t - 1.) * eps * noise_level
        if not skip_projection:
            x_adv.clamp_(0., 1.)
    
    criterion_indiv = criterion_dict[loss]

    x_adv.requires_grad = True
    logits = model(x_adv)
    loss_indiv = criterion_indiv(logits, y)
    grad = torch.autograd.grad(loss_indiv.sum(), x_adv)[0].detach()
    
    x_adv = x_adv.detach() + alpha * eps * grad.sign()
    if not skip_projection:
        x_adv = x + (x_adv - x).clamp(-eps, eps)
        x_adv.clamp_(0., 1.)
    
    return x_adv
