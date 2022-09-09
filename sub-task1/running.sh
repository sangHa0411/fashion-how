python train.py \
    --learning_rate 5e-5 \
    --epochs 3 \
    --backbone resnet \
    --loss softmax \
    --num_model 1 \
    --warmup_ratio 0.05 \
    --num_aug 5 \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4

python train.py \
    --learning_rate 5e-5 \
    --epochs 3 \
    --backbone densenet \
    --loss arcface \
    --num_model 2 \
    --warmup_ratio 0.05 \
    --num_aug 5 \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4

python train.py \
    --learning_rate 5e-5 \
    --epochs 3 \
    --backbone vgg \
    --loss rdrop \
    --num_model 3 \
    --warmup_ratio 0.05 \
    --num_aug 5 \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4
