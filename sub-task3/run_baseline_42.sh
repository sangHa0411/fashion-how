# STEP 1
python train.py --in_file_trn_dialog ./data/task1.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --learning_rate 2e-5 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42_42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

# STEP 2
python train.py --in_file_trn_dialog ./data/task2.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --learning_rate 1e-5 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

# STEP 3
python train.py --in_file_trn_dialog ./data/task3.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --learning_rate 5e-6 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

# STEP 4
python train.py --in_file_trn_dialog ./data/task4.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --learning_rate 3e-6 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

# STEP 5
python train.py --in_file_trn_dialog ./data/task5.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --learning_rate 2e-6 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

# STEP 6
python train.py --in_file_trn_dialog ./data/task6.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --seed 42 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --learning_rate 1e-6 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final-42.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --text_feat_size 512 \
    --img_feat_size 512 \
    --eval_node [6000,6000,6000,6000,512][2000,2000,2000]