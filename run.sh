python src/train.py --patch_size=64 \
                    --batch_size=32 \
                    --max_steps=300000 \
                    --decay=200000 \
                    --model=resnet \
                    --reduce_upsample=True \
                    --group=4 \
                    --ckpt_name=resnet-4-ru \
                    --scale=2 --num_gpu=2
