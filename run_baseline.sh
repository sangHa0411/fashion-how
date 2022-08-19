# STEP 1
python train.py --in_file_trn_dialog ./data/task1.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

# STEP 2
python train.py --in_file_trn_dialog ./data/task2.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

# STEP 3
python train.py --in_file_trn_dialog ./data/task3.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

# STEP 4
python train.py --in_file_trn_dialog ./data/task4.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

# STEP 5
python train.py --in_file_trn_dialog ./data/task5.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

# STEP 6
python train.py --in_file_trn_dialog ./data/task6.ddata.wst.txt \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --epochs 10 \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --dropout_prob 0.1 \
    --batch_size 16 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --logging_steps 300 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task1.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task2.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]

python evaluate.py --in_file_tst_dialog ./data/cl_eval_task3.wst.dev \
    --subWordEmb_path ./data/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
    --model_path gAIa_CL_model \
    --model_file gAIa-final.pt \
    --dropout_prob 0.1 \
    --eval_batch_size 8 \
    --hops 3 \
    --mem_size 24 \
    --key_size 512 \
    --img_feat_size 512 \
    --text_feat_size 1024 \
    --eval_node [4096,4096,4096,4096,1024][2048,2048,2048]