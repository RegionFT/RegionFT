cd ../table_model

root='../exp/'


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_g_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&g_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&r_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&r_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&r_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&r_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r&g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r&g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r&g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r&g_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_g_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&g_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&r_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&r_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&r_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&r_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r&g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r&g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r&g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r&g_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_g_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&g_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_a&r_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_a&r_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_a&r_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_a&r_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/adult_r&g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/adult_r&g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/adult_r&g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/adult_r&g_0.5_predict.log'  





# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_a_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_a_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_a_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_a_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g_1_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g&a_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g&a_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g&a_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g&a_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_a_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_a_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_a_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_a_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g_0.01_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g&a_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g&a_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g&a_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g&a_0.01_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_a_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_a_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_a_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_a_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g_0.5_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/german_g&a_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/german_g&a_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/german_g&a_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/german_g&a_0.5_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/bank_a_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/bank_a_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/bank_a_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/bank_a_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/bank_a_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/bank_a_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/bank_a_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/bank_a_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/bank_a_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/bank_a_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/bank_a_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/bank_a_0.01_predict.log'  




# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_g_1_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r_1_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r&g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r&g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r&g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r&g_1_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_g_0.01_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r_0.01_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r&g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r&g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r&g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r&g_0.01_predict.log'  



# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_g_0.5_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r_0.5_predict.log'  

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/compas_r&g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/compas_r&g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/compas_r&g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/compas_r&g_0.5_predict.log'  




# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/lsac_g_1.h5' \
# --output_path=$root'train_dnn/faire_quant_models/lsac_g_1/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/lsac_g_1/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/lsac_g_1_predict.log'  


CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/faire_quant_models/lsac_r_1.h5' \
--output_path=$root'train_dnn/faire_quant_models/lsac_r_1/predict_scores.npy' \
--output_path2=$root'train_dnn/faire_quant_models/lsac_r_1/labels.npy' \
2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r_1_predict.log'  

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/faire_quant_models/lsac_r&g_1.h5' \
--output_path=$root'train_dnn/faire_quant_models/lsac_r&g_1/predict_scores.npy' \
--output_path2=$root'train_dnn/faire_quant_models/lsac_r&g_1/labels.npy' \
2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r&g_1_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/lsac_g_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/lsac_g_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/lsac_g_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/lsac_g_0.01_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/lsac_r_0.01.h5' \
# --output_path=$root'train_dnn/faire_quant_models/lsac_r_0.01/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/lsac_r_0.01/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r_0.01_predict.log'  

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/faire_quant_models/lsac_r&g_0.01.h5' \
--output_path=$root'train_dnn/faire_quant_models/lsac_r&g_0.01/predict_scores.npy' \
--output_path2=$root'train_dnn/faire_quant_models/lsac_r&g_0.01/labels.npy' \
2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r&g_0.01_predict.log'  



# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=census \
# --dataset_path=$root'table/census/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/lsac_g_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/lsac_g_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/lsac_g_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/lsac_g_0.5_predict.log'  


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/faire_quant_models/lsac_r_0.5.h5' \
# --output_path=$root'train_dnn/faire_quant_models/lsac_r_0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/faire_quant_models/lsac_r_0.5/labels.npy' \
# 2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r_0.5_predict.log'  

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
--dataset_path=$root'table/lsac/sampled_data.csv' \
--model_path=$root'train_dnn/faire_quant_models/lsac_r&g_0.5.h5' \
--output_path=$root'train_dnn/faire_quant_models/lsac_r&g_0.5/predict_scores.npy' \
--output_path2=$root'train_dnn/faire_quant_models/lsac_r&g_0.5/labels.npy' \
2>&1 | tee $root'train_dnn/faire_quant_models/lsac_r&g_0.5_predict.log'  