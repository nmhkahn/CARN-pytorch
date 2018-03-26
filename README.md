
# Fast, Accurate, and, Lightweight Super-Resolution with Cascading Residual Network
Namhyuk Ahn, Byungkon Kang, Kyung-Ah Sohn [[Arxiv]](https://arxiv.org/abs/1803.08664)

### Abstract
In recent years, deep learning methods have been successfully applied to single-image super-resolution tasks. Despite their great performances, deep learning methods cannot be easily applied to real-world applications due to the requirement of heavy computation. In this paper, we address this issue by proposing an accurate and lightweight deep learning model for image super-resolution. In detail, we design an architecture that implements a cascading mechanism upon a residual network. We also present a variant model of the proposed cascading residual network to further improve efficiency. Our extensive experiments show that even with much fewer parameters and operations, our models achieve performance comparable to that of state-of-the-art methods. 

### Requirements
- Python 3
- [PyTorch](https://github.com/pytorch/pytorch) (0.3.0), [torchvision](https://github.com/pytorch/vision)
- Numpy, Scipy
- Pillow, Scikit-image
- h5py
- importlib

### Dataset
We use DIV2K dataset for training and Set5, Set14, B100 and Urban100 dataset for the benchmark test. Here are the following steps to prepare datasets.

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) and unzip on `dataset` directory as below:
  ```
  dataset
  └── DIV2K
      ├── DIV2K_train_HR
      ├── DIV2K_train_LR_bicubic
      ├── DIV2K_valid_HR
      └── DIV2K_valid_LR_bicubic
  ```
2. To accelerate training, we first convert training images to h5 format as follow (h5py module has to be installed).
```shell
$ cd datasets && python div2h5.py
```
3. Other benchmark datasets can be downloaded in [here](https://drive.google.com/file/d/1JJFKMRdOF4DqZd1kwDRrPKvSKnmqWjed/view?usp=sharing). Same as DIV2K, please put all the datasets in `dataset` directory.

### Test Pretrained Models
We provide the pretrained models in `checkpoint` directory. To test CARN on benchmark dataset:
```shell
$ python carn/sample.py --model carn \
                        --test_data_dir ... \
                        --scale 4 \
                        --ckpt_path ./checkpoint/carn.pth \
                        --sample_dir ./sample
```
and for CARN-M,
```shell
$ python carn/sample.py --model carn_m \
                        --test_data_dir ... \
                        --scale 4 \
                        --group 4 \
                        --reduce_upsample \
                        --ckpt_path ./checkpoint/carn_m.pth \
                        --sample_dir ./sample \
```
To test on DIV2K dataset, set `test_data_dir` argument as `dataset/DIV2K/DIV2K_valid` and for other dataset, it should be `dataset/other_dataset_dir`.
Or, we provide our results on four benchmark dataset (Set5, Set14, B100 and Urban100) in [here](https://drive.google.com/file/d/1RGio4rgo1f8vjUJlp891gRqY8Fov40hD/view?usp=sharing).

### Training Models
Here are our default settings to train CARN and CARN-M. Note: We only have experiments on TITAN X or 1080ti. If OOM error occurs, please decrease batch size.
```shell
# For CARN
python carn/train.py --patch_size 48 \
                     --batch_size 32 \
                     --max_steps 500000 \
                     --decay 300000 \
                     --model carn \
                     --ckpt_name carn \
                     --scale 0 \
                     --num_gpu 1
# For CARN-M
python carn/train.py --patch_size 48 \
                     --batch_size 32 \
                     --max_steps 500000 \
                     --decay 300000 \
                     --model carn_m \
                     --ckpt_name carn_m \
                     --group 4 \
                     --reduce_upsample \
                     --scale 0 \
                     --num_gpu 1
```
In the `--scale` argument, [2, 3, 4] is for single-scale training and 0 for multi-scale learning. `--group` represents group size of group convolution inside of efficient residual block. `--reduce_upsample` means use 1x1 convolution instead of 3x3 in the upsampling layer.

### Results
Quantitative evaluation of state-of-the-art SR algorithms
<img src="assets/sota.png" width="70%">

Visual qualitative comparison on 4× scale datasets.
<img src="assets/result.png" width="70%">

### Citation
```
@article{ahn2018fast,
  title={Fast, Accurate, and, Lightweight Super-Resolution with Cascading Residual Network},
  author={Ahn, Namhyuk and Kang, Byungkon and Sohn, Kyung-Ah},
  journal={arXiv preprint arXiv:1803.08664},
  year={2018}
}
```
