import os
import numpy as np
import pandas as pd

# make predictions for 1 to 6
os.system('cd /home/wkrtkd911/project/fashion-how')
os.system('sh run_pred.sh --data_order 1 --in_file_tst_dialog data/test/cl_eval_task1.wst.tst --model_path ./gAIa_CL_model')
os.system('sh run_pred.sh --data_order 2 --in_file_tst_dialog data/test/cl_eval_task2.wst.tst --model_path ./gAIa_CL_model')
os.system('sh run_pred.sh --data_order 3 --in_file_tst_dialog data/test/cl_eval_task3.wst.tst --model_path ./gAIa_CL_model')
os.system('sh run_pred.sh --data_order 4 --in_file_tst_dialog data/test/cl_eval_task4.wst.tst --model_path ./gAIa_CL_model')
os.system('sh run_pred.sh --data_order 5 --in_file_tst_dialog data/test/cl_eval_task5.wst.tst --model_path ./gAIa_CL_model')
os.system('sh run_pred.sh --data_order 6 --in_file_tst_dialog data/test/cl_eval_task6.wst.tst --model_path ./gAIa_CL_model')


# combine prediction results
preds = []
for i in range(1, 7):
    preds.append(pd.read_csv(f"./predictions/prediction{i}.csv", header=None, encoding='utf8').to_numpy())
preds = np.concatenate(preds, axis=0)
np.savetxt("./predictinosprediction.csv", preds.astype(int), encoding='utf8', fmt='%d')

# remove remnants
for i in range(1, 7):
    os.remove(f"./predictions/prediction{i}.csv")