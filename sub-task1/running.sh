# Training & Evaluating
python train.py \
    --learning_rate 5e-5 \
    --epochs 5 \
    --loss softmax \
    --num_model 1 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --do_eval True \
    --num_aug 0 \
    --hidden_size 2048 \
    --batch_size 128 \
    --eval_batch_size 16 \
    --img_size 224 \
    --weight_decay 1e-3 \
    --logging_steps 10 \
    --save_steps 30 \
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