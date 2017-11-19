import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_name", type=str)
    
    parser.add_argument("--train_data_path", type=str, 
                        default="dataset/DIV2K_train.h5")
    parser.add_argument("--test_data_dir", type=str, 
                        default="dataset/DIV2K_valid")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoint")
    
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)

    parser.add_argument("--verbose", action="store_true", default="store_true")

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--decay", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)

    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1"], default="L1")

    return parser.parse_args()

def main(cfg):
    if cfg.scale == 0:
        cfg.scale = [2, 3, 4]

    # dynamic import using --model argument
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    trainer = Trainer(net, cfg)
    trainer.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
