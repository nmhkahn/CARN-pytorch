python src/train.py --patch_size=96 \
                    --batch_size=32 \
                    --max_steps=300000 \
                    --decay=200000 \
                    --model=mdrn \
                    --ckpt_name=mdrnet-A \
                    --num_gpu=2 \
                    --scale=0
