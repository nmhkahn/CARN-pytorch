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
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_gpu", type=int,
                        default=1)
    parser.add_argument("--scale", type=int,
                        default=2)

    return parser.parse_args()

def main(config):
    if config.model in ["mdrn", "base", "mdrn_multi", "mdrn_multi_v2"]:
        config.max_epoch = 50
        config.batch_size = 32
        config.patch_size = 64
        config.lr = 0.0001
        config.step = 30
        config.clip = 0.4
        config.verbose = True
    
    print(json.dumps(vars(config), indent=4, sort_keys=True))
    trainer = Trainer(config)
    trainer.fit()

if __name__ == "__main__":
    config = parse_args()
    main(config)
