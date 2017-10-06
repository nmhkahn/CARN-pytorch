import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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
                        default=4)

    return parser.parse_args()

def main(config):
    if config.model in ["mdrn"]:
       config.batch_size = 32
       config.patch_size = 96*config.scale
    
    lr_list = [0.01, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
    for lr in lr_list:
        config.lr = lr
        config.decay = 10000
        config.max_epoch = 500
        config.verbose = False

        trainer = Trainer(config)
        trainer.fit()
        psnr = trainer.evaluate()
        print(lr, psnr)

if __name__ == "__main__":
    config = parse_args()
    main(config)
