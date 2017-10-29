import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--patch_size", type=int)
    
    parser.add_argument("--train_data_path", type=str, 
                        default="dataset/DIV2K_train.h5")
    parser.add_argument("--test_data_dir", type=str, 
                        default="dataset/DIV2K_valid")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint")
    
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)

    return parser.parse_args()

def main(cfg):
    if cfg.model in ["vdsr", "base"]:
        if cfg.model in ["vdsr"]:
            from model.vdsr import Net
        elif cfg.model in ["base"]:
            from model.base import Net
    elif cfg.model in ["dnet", "rnet"]:
        if cfg.model in ["dnet"]:
            from model.dnet import Net
        elif cfg.model in ["rnet"]:
            from model.rnet import Net
    
    # common settings
    cfg.max_steps = 100000
    cfg.batch_size = 64
    cfg.lr = 0.0001
    cfg.clip = 0.4
    cfg.decay = 30000
    cfg.verbose = True
            
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    trainer = Trainer(Net, cfg)
    trainer.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
