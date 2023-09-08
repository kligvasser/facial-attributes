import copy
import logging
import torch
from ast import literal_eval

from torchvision import models as torch_models
from models.models import TransferModel


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_model_entire(model, path):
    torch.save(model, path)


def load_model(args, path=None):
    if args.model_config != '':
        model_config = dict({}, **literal_eval(args.model_config))
    else:
        model_config = {}

    model = init_model(**model_config)

    if path:
        logging.info('Loading model from {}...'.format(path))
        model.load_state_dict(torch.load(path))

    return model


def load_model_entire(path):
    model = torch.load(path)
    model.eval()

    return model


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def init_model(**config):
    config.setdefault('num_classes', 40)

    model = config.pop('model', 'resnet50')
    if model == 'resnet34':
        return TransferModel(backbone=torch_models.resnet34(pretrained=True), **config)
    elif model == 'resnet50':
        return TransferModel(backbone=torch_models.resnet50(pretrained=True), **config)
    elif model == 'resnet101':
        return TransferModel(backbone=torch_models.resnet101(pretrained=True), **config)
    elif model == 'mobilenetv2':
        return TransferModel(backbone=torch_models.mobilenet_v2(pretrained=True), **config)
    else:
        raise ValueError('Unsupported model: {}.'.format(model))
