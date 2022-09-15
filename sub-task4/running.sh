# Seed 42
python train.py --in_file_trn_dialog data/ddata.wst.txt.2021.6.9 \
    --in_file_fashion data/mdata.txt.2021.10.18 \
    --subWordEmb_path data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --img_feat_dir data/img_feats \
    --model_path /home/wkrtkd911/project/fashion-how/sub-task4/gAIa_model \
    --seed 42 \
    --epochs 15 \
    --learning_rate 5e-5 \
    --dropout_prob 0.1 \
    --batch_size 128 \
    --num_layers 6 \
    --d_model 512 \
    --hidden_size 2048 \
    --num_head 8 \
    --hops 3 \
    --augmentation_size 5 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --logging_steps 500 \
    --save_steps 1000 \
    --eval_node [1024,4096,4096]

# Seed 1234
python train.py --in_file_trn_dialog data/ddata.wst.txt.2021.6.9 \
    --in_file_fashion data/mdata.txt.2021.10.18 \
    --subWordEmb_path data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --img_feat_dir data/img_feats \
    --model_path /home/wkrtkd911/project/fashion-how/sub-task4/gAIa_model \
    --seed 1234 \
    --epochs 20 \
    --learning_rate 5e-5 \
    --dropout_prob 0.1 \
    --batch_size 128 \
    --num_layers 6 \
    --d_model 512 \
    --hidden_size 2048 \
    --num_head 8 \
    --hops 3 \
    --augmentation_size 5 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --logging_steps 500 \
    --save_steps 1000 \
    --eval_node [1024,4096,4096]

# Seed 95
python train.py --in_file_trn_dialog data/ddata.wst.txt.2021.6.9 \
    --in_file_fashion data/mdata.txt.2021.10.18 \
    --subWordEmb_path data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --img_feat_dir data/img_feats \
    --model_path /home/wkrtkd911/project/fashion-how/sub-task4/gAIa_model \
    --seed 95 \
    --epochs 20 \
    --learning_rate 5e-5 \
    --dropout_prob 0.1 \
    --batch_size 128 \
    --num_layers 6 \
    --d_model 512 \
    --hidden_size 2048 \
    --num_head 8 \
    --hops 3 \
    --augmentation_size 5 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --logging_steps 500 \
    --save_steps 1000 \
    --eval_node [1024,4096,4096]