scale=2
ckpt_path="model/Base-64W-128S.pth"
model="base"

python src/sample.py --model=$model --ckpt_path=$ckpt_path --sample_dir=sample/ --test_data_dir=dataset/Set14 --cuda --scale=$scale
python src/sample.py --model=$model --ckpt_path=$ckpt_path --sample_dir=sample/ --test_data_dir=dataset/Set5 --cuda --scale=$scale
python src/sample.py --model=$model --ckpt_path=$ckpt_path --sample_dir=sample/ --test_data_dir=dataset/BSD100 --cuda --scale=$scale
