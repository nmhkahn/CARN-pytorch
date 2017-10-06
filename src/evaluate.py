import os
import json
from collections import defaultdict, OrderedDict
import argparse
import numpy as np
import skimage.measure as measure

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import FlickrDataset

TYPE = {"gwn": 1, "snp": 2, "quant": 3, "jpeg": 4,
        "gblur": 5, "denoising": 6, "low_res": 7, "fnoise": 8}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_data_path", type=str, 
                        default="flickr/test_gray.json")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", action="store_true")

    return parser.parse_args()


def evaluate(refiner, config):
    dataset = FlickrDataset(config.test_data_path, -1)
    with open(config.test_data_path) as _file:
        infos = json.load(_file)
    
    result = defaultdict(lambda: [0, 0, 0]) # count, psnr, ssim

    for step in range(len(dataset)):
        inputs = dataset[step]
        reference = Variable(inputs[0].unsqueeze(0), volatile=True)
        distorted = Variable(inputs[1].unsqueeze(0), volatile=True)
        objects   = infos[step]["objects"]

        if config.cuda:
            reference = reference.cuda()
            distorted = distorted.cuda()

        refined = refiner(distorted)

        if config.save:
            torchvision.utils.save_image(distorted.data,
                "sample/eval_{}_distorted.png".format(step))
            torchvision.utils.save_image(refined.data,
                "sample/eval_{}_refined.png".format(step))

        refined = refined.data.cpu().numpy()[0]
        reference = reference.data.cpu().numpy()[0]
        
        reference = np.squeeze(reference, 0)
        refined   = np.squeeze(refined, 0)
            
        reference = (np.clip(reference, 0, 1)*255).astype(np.uint8)
        refined   = (np.clip(refined, 0, 1)*255).astype(np.uint8)
        
        # eval distorted region
        for obj in objects:
            im1, im2 = reference.copy(), reference.copy()
            
            tp = obj["type"]
            xmin, xmax = int(obj["xmin"]), int(obj["xmax"])
            ymin, ymax = int(obj["ymin"]), int(obj["ymax"])
            
            im1 = im1[ymin:ymax, xmin:xmax]
            im2 = refined[ymin:ymax, xmin:xmax]
            
            result[tp][0] += 1
            result[tp][1] += measure.compare_psnr(im1, im2)
            result[tp][2] += measure.compare_ssim(im1, im2)

        # eval entire image 
        result["total"][0] += 1
        result["total"][1] += measure.compare_psnr(reference, refined)
        result["total"][2] += measure.compare_ssim(reference, refined)
        
    return result


def main(config):
    model_name = "_".join(config.model_path.split("/")[-1].split("_")[:-1])

    if model_name in ["dncnn"]:
        from model.dncnn import DnCNN

        refiner = DnCNN()
    elif model_name in ["residual", "residual_v2", "residual_v3"]:
        if model_name in ["residual"]:
            from model.residual import Residual
        elif model_name in ["residual_v2"]:
            from model.residual_v2 import Residual
        else:
            from model.residual_v3 import Residual
        refiner = Residual()
    else:
        raise NotImplementedError("{} is not in our list".format(model_name))
    
    state_dict = torch.load(config.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    refiner.load_state_dict(new_state_dict)

    if config.cuda:
        refiner.cuda()
    result = evaluate(refiner, config)
    
    print("{:10}{:10}{:10}".format("", "PSNR", "SSIM"))
    for k, v in result.items():
        print("{:10}{:10.3f}{:10.3f}".format(k, v[1]/v[0], v[2]/v[0]))


if __name__ == "__main__":
    config = parse_args()
    main(config)
