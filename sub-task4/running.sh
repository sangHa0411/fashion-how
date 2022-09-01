python train.py --in_file_trn_dialog data/ddata.wst.txt.2021.6.9 \
    --in_file_fashion data/mdata.txt.2021.10.18 \
    --subWordEmb_path data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --img_feat_dir data/img_feats \
    --model_path gAIa_model \
    --epochs 5 \
    --learning_rate 5e-5 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --augmentation_size 2 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]