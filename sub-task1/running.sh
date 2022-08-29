python train.py \
    --learning_rate 5e-5 \
    --epochs 5 \
    --num_aug 5 \
    --batch_size 128 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-3 \
    --logging_steps 100 \
    --save_steps 500 \
    --num_workers 4