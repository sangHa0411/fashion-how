# Training & Evaluating
# softmax
python train.py \
    --learning_rate 5e-5 \
    --epochs 5 \
    --loss softmax \
    --num_model 1 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --do_eval True \
    --num_aug 3 \
    --hidden_size 2048 \
    --batch_size 128 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-3 \
    --logging_steps 50 \
    --save_steps 100 \
    --num_workers 4

# arcface
python train.py \
    --learning_rate 5e-5 \
    --epochs 5 \
    --loss arcface \
    --num_model 1 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --do_eval True \
    --num_aug 3 \
    --hidden_size 2048 \
    --batch_size 128 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-3 \
    --logging_steps 50 \
    --save_steps 100 \
    --num_workers 4

# rdrop
python train.py \
    --learning_rate 5e-5 \
    --epochs 5 \
    --loss rdrop \
    --num_model 1 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --do_eval True \
    --num_aug 3 \
    --hidden_size 2048 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-3 \
    --logging_steps 100 \
    --save_steps 300 \
    --num_workers 4

# # Full Training
# python train.py \
#     --learning_rate 5e-5 \
#     --epochs 5 \
#     --num_model 1 \
#     --warmup_ratio 0.05 \
#     --num_aug 3 \
#     --batch_size 64 \
#     --eval_batch_size 16 \
#     --img_size 224 \
#     --weight_decay 1e-3 \
#     --logging_steps 100 \
#     --save_steps 300 \
#     --num_workers 4

# python train.py \
#     --learning_rate 5e-5 \
#     --epochs 5 \
#     --seed 1 \
#     --num_model 2 \
#     --warmup_ratio 0.05 \
#     --num_aug 3 \
#     --batch_size 64 \
#     --eval_batch_size 16 \
#     --img_size 224 \
#     --weight_decay 1e-3 \
#     --logging_steps 100 \
#     --save_steps 300 \
#     --num_workers 4

# python train.py \
#     --learning_rate 5e-5 \
#     --epochs 5 \
#     --seed 1234 \
#     --num_model 3 \
#     --warmup_ratio 0.05 \
#     --num_aug 3 \
#     --batch_size 64 \
#     --eval_batch_size 16 \
#     --img_size 224 \
#     --weight_decay 1e-3 \
#     --logging_steps 100 \
#     --save_steps 300 \
#     --num_workers 4