import os
import json
import math
import requests
import torch
from collections import OrderedDict
from model_zoo.models import model_dicts
#from models_new import l_models_all, l_models_imagenet

def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    print('Download finished: path={} (gdrive_id={})'.format(fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def load_model(model_name, model_dir='./models', norm='Linf'):
    model_dir_norm = '{}/{}'.format(model_dir, norm)
    if not isinstance(model_dicts[norm][model_name]['gdrive_id'], list):
        model_path = '{}/{}/{}.pt'.format(model_dir, norm, model_name)
        model = model_dicts[norm][model_name]['model']()
        if not os.path.exists(model_dir_norm):
            os.makedirs(model_dir_norm)
        if not os.path.isfile(model_path):
            download_gdrive(model_dicts[norm][model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location='cuda:0')
    
        # needed for the model of `Carmon2019Unlabeled`
        try:
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
        
        model.load_state_dict(state_dict, strict=True)
        return model.cuda().eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model_path = '{}/{}/{}'.format(model_dir, norm, model_name)
        model = model_dicts[norm][model_name]['model']()
        if not os.path.exists(model_dir_norm):
            os.makedirs(model_dir_norm)
        for i, gid in enumerate(model_dicts[norm][model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i), map_location='cuda:0')
            try:
                state_dict = rm_substr_from_state_dict(checkpoint['state_dict'], 'module.')
            except:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            model.models[i].load_state_dict(state_dict)
            model.models[i].cuda().eval()
        return model


def clean_accuracy(model, x, y, batch_size=100):
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].cuda()

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def get_accuracy_and_logits(model, x, y, batch_size=100, n_classes=10):
    logits = torch.zeros([y.shape[0], n_classes])
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].cuda()
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].cuda()

            output = model(x_curr)
            logits[counter * batch_size:(counter + 1) * batch_size] += output.cpu()
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0], logits



def list_available_models(norm='Linf'):
    models = model_dicts[norm].keys()

    json_dicts = []
    for model_name in models:
        with open('./model_info/{}.json'.format(model_name), 'r') as model_info:
            json_dict = json.load(model_info)
        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict['venue'] == '' else json_dict['venue']
        json_dict['AA'] = float(json_dict['AA']) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -d['AA'])
    print('| # | Model ID | Paper | Clean accuracy | Robust accuracy | Architecture | Venue |')
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        print('| <sub>**{}**</sub> | <sub>**{}**</sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'.format(
            i+1, json_dict['model_name'], json_dict['name'], json_dict['link'], json_dict['clean_acc'], json_dict['AA'],
            json_dict['architecture'], json_dict['venue']))


def load_model_fast_at(model_name, norm, model_dir, fts_before_bn):
    from model_zoo.fast_models import PreActResNet18, model_names
    model_name_long = model_names[norm][model_name]
    activation = [c.split('activation=')[-1] for c in model_name_long.split(' ') if 'activation' in c]
    if 'normal=' in model_name_long:
        normal = model_name_long.split('normal=')[1].split(' ')[0]
    else:
        normal = 'none'
    
    if 'resnet18' in model_name_long:
        model = PreActResNet18(n_cls=10, activation=activation[0], fts_before_bn=fts_before_bn,
            normal=normal)
        ckpt = torch.load('{}/{}'.format(model_dir, model_name_long))
        model.load_state_dict({k: v for k, v in ckpt['last'].items() if 'model_preact_hl1' not in k})
    return model.eval()


def load_model_ssl(model_name, model_dir):
    from model_zoo.ssl_models import models_dict
    data = models_dict[model_name]
    model = data['model']()
    ckpt_base = torch.load('{}/{}'.format(model_dir, data['base']))['model']
    ckpt_base = rm_substr_from_state_dict(ckpt_base, 'module.')
    model.base.load_state_dict(ckpt_base)
    ckpt_lin = torch.load('{}/{}'.format(model_dir, data['linear']))['model']
    ckpt_lin = rm_substr_from_state_dict(ckpt_lin, 'module.')
    model.linear.load_state_dict(ckpt_lin)
    model.cuda()
    model.eval()
    return model

def load_anymodel(model_name, model_dir='./models'):
    if len(model_name) == 2:
        return load_model(model_name[0], model_dir='./models', norm=model_name[1]).cuda().eval()
    elif len(model_name) == 3 and model_name[2] == 'fast_at':
        return load_model_fast_at(model_name[0], model_name[1],
            model_dir=model_dir, #'./models' #'../understanding-fast-adv-training-dev/models'
            fts_before_bn=False).cuda().eval()
    elif len(model_name) == 3 and model_name[2] == 'ssl':
        model = load_model_ssl(model_name[0], './models/ssl_models')
        assert not model.base.training
        assert not model.linear.training
        return model
    elif len(model_name) == 3 and model_name[2] == 'ext':
        from model_zoo.ext_models import load_ext_models
        return load_ext_models(model_name[0])


def load_anymodel_imagenet(model_name, **kwargs):
    if len(model_name) == 2:
        from model_zoo.models_imagenet import load_model as load_model_imagenet
        return load_model_imagenet(model_name[0], norm=model_name[1])
    elif len(model_name) == 3 and model_name[2] == 'pretrained':
        from model_zoo.models_imagenet import PretrainedModel
        model = PretrainedModel(model_name[0])
        assert not model.model.training
        return model
    elif len(model_name) == 3 and model_name[2] == 'ssl':
        model = load_model_ssl(model_name[0], './models/ssl_models')
        assert not model.base.training
        assert not model.linear.training
        return model
    elif len(model_name) == 3 and model_name[2] == 'ext':
        from model_zoo.ext_models import load_ext_models_imagenet
        return load_ext_models_imagenet(model_name[0], **kwargs)

def load_anymodel_cifar100(model_name):
    if len(model_name) == 2:
        from model_zoo.models_cifar100 import load_model as load_model_cifar100
        return load_model_cifar100(model_name[0], norm=model_name[1])


def load_anymodel_imagenet100(model_name):
    if len(model_name) == 3 and model_name[2] == 'ext':
        from model_zoo.ext_models import load_ext_models_imagenet100
        return load_ext_models_imagenet100(model_name[0], model_name[1])
        

def load_anymodel_mnist(model_name):
    if len(model_name) == 2:
        from model_zoo.models_mnist import load_model as load_model_mnist
        return load_model_mnist(model_name[0], norm=model_name[1])


'''def load_anymodel_datasets(args):
    fts_idx = [int(c) for c in args.fts_idx.split(' ')]
    if args.dataset == 'cifar10':
        l_models = [l_models_all[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel(l_models[0])
        model.eval()
    elif args.dataset == 'imagenet':
        l_models = [l_models_imagenet[c] for c in fts_idx]
        print(l_models)
        model = load_anymodel_imagenet(l_models[0])
        #sys.exit()
        with torch.no_grad():
            acc = clean_accuracy(model, x, y, batch_size=25)
        print('clean accuracy: {:.1%}'.format(acc))
    return model'''

if __name__ == '__main__':
    #list_available_models()
    pass
