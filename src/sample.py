import os
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--test_data_dir", type=str, default="dataset/Set5")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--self_ensemble", action="store_true")

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()[0]
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def sample(refiner, dataset, config):
    for step, (hr, lr) in enumerate(dataset):
        hr = Variable(hr.unsqueeze(0), volatile=True)
        lr = Variable(lr.unsqueeze(0), volatile=True)

        if config.cuda:
            hr = hr.cuda()
            lr = lr.cuda()

        sr = refiner(lr)

        sample_dir = os.path.join(config.sample_dir,
            config.test_data_dir.split("/")[-1])
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        model_name = config.ckpt_path.split(".")[0].split("/")[-1]

        save_image(sr.data,
            os.path.join(sample_dir, "{}_{}_SR.png".format(model_name, step)))
        save_image(hr.data,
            os.path.join(sample_dir, "{}_{}_HR.png".format(model_name, step)))


def main(config):
    if config.model in ["mdrn"]:
        from model.mdrn import MDRN
        refiner = MDRN(config.scale)
    elif config.model in ["base"]:
        from model.base import MDRN
        refiner = MDRN(config.scale)
    elif config.model in ["mdrn_v2"]:
        from model.mdrn_v2 import MDRN
        refiner = MDRN(config.scale) 
    else:
        raise NotImplementedError("{} is not in our list".format(model_name))
    
    state_dict = torch.load(config.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v
    refiner.load_state_dict(new_state_dict)

    if config.cuda:
        refiner.cuda()
    
    dataset = TestDataset(config.test_data_dir, 
                          config.scale, 
                          config.self_ensemble)
    sample(refiner, dataset, config)
 

if __name__ == "__main__":
    config = parse_args()
    main(config)
