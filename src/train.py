import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_data_path", type=str, 
                        default="dataset/DIV2K_train")
    parser.add_argument("--test_data_path", type=str, 
                        default="dataset/Set14")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_gpu", type=int,
                        default=2)
    parser.add_argument("--scale", type=int,
                        default=2)

    return parser.parse_args()

def main(config):
    if config.model in ["mdrn"]:
        config.max_epoch = 3000
        config.decay = 2000
        config.batch_size = 64
        config.patch_size = 48*config.scale
        config.lr = 0.0001
        config.verbose = True
    if config.model in ["edsr"]:
        config.max_epoch = 3000
        config.decay = 2000
        config.batch_size = 64
        config.patch_size = 48*config.scale
        config.lr = 0.0001
        config.verbose = True
    
    trainer = Trainer(config)
    print(json.dumps(vars(config), indent=4, sort_keys=True))
    trainer.fit()

if __name__ == "__main__":
    config = parse_args()
    main(config)
