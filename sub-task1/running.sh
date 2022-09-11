python train.py \
    --seed 42 \
    --learning_rate 1e-4 \
    --epochs 3 \
    --backbone ResNetFeedForwardModel \
    --loss softmax \
    --num_model 1 \
    --num_aug 4 \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4


python train.py \
    --seed 12 \
    --learning_rate 1e-4 \
    --epochs 3 \
    --backbone ResNetFeedForwardModel \
    --loss softmax \
    --num_model 2 \
    --num_aug 5 \
    --do_eval True \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4


python train.py \
    --seed 412 \
    --learning_rate 1e-4 \
    --epochs 3 \
    --backbone ResNetFeedForwardModel \
    --loss softmax \
    --num_model 3 \
    --num_aug 5 \
    --do_eval True \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-4 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4

