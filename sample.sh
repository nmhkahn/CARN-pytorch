model=
scale=

python src/sample.py --model=$model --ckpt_path=model/$model.pth --sample_dir=sample/ --test_data_dir=dataset/Set14 --scale=$scale
python src/sample.py --model=$model --ckpt_path=model/$model.pth --sample_dir=sample/ --test_data_dir=dataset/Set5 --scale=$scale
python src/sample.py --model=$model --ckpt_path=model/$model.pth --sample_dir=sample/ --test_data_dir=dataset/B100 --scale=$scale
