cd ../table_model

root='../exp/'

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/bank.h5' \
# --output_path=$root'train_dnn/multitask_models/bank/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/bank/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/multitask_models/bank/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/bank_model_a.h5' \
# --output_path=$root'train_dnn/models_flip/bank_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/bank_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_flip/bank_model_a/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/bank_model_a.h5' \
# --output_path=$root'train_dnn/models_adv/bank_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/bank_model_a/labels.npy' \
# 2>&1 | tee ../exp/train_dnn/models_adv/bank_model_a/predict.log

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3.h5' \
# --output_path=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/bank_a_gated_4_0.3_0.2_p-0.15_p0.3/predict.log'   &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/retrained_models/bank_EIDIG_INF_retrained_model/predict.log'   &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=bank \
# --dataset_path=$root'table/bank/sampled_data.csv' \
# --model_path=$root'train_dnn/train/bank_model.h5' \
# --output_path=$root'train_dnn/train/bank_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/bank_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/train/bank_model/predict.log'   &



# CUDA_VISIBLE_DEVICES=1 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/german_model_g.h5' \
# --output_path=$root'train_dnn/models_flip/german_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/german_model_g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/german_model_g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=1 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/german_model_a.h5' \
# --output_path=$root'train_dnn/models_flip/german_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/german_model_a/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/german_model_a/predict.log'   & 

# CUDA_VISIBLE_DEVICES=1 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/german_model_g&a.h5' \
# --output_path=$root'train_dnn/models_flip/german_model_g&a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/german_model_g&a/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/german_model_g&a/predict.log'   & 


# CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/german_model_0.8_a.h5' \
# --output_path=$root'train_dnn/models_adv/german_model_0.8_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/german_model_0.8_a/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/german_model_0.8_a/predict.log'   & 
CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/models_adv/german_model_0.95_g&a.h5' \
--output_path=$root'train_dnn/models_adv/german_model_0.95_g&a/predict_scores.npy' \
--output_path2=$root'train_dnn/models_adv/german_model_0.95_g&a/labels.npy' \
2>&1 | tee '../exp/train_dnn/models_adv/german_model_0.95_g&a/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/german_model_a.h5' \
# --output_path=$root'train_dnn/models_adv/german_model_a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/german_model_a/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/german_model_a/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/german_model_g&a.h5' \
# --output_path=$root'train_dnn/models_adv/german_model_g&a/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/german_model_g&a/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/german_model_g&a/predict.log'   & 

# CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/german_model.h5' \
# --output_path=$root'train_dnn/multitask_models/german_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/german_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/multitask_models/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/german_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/german_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/german_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/retrained_models/german_EIDIG_INF_retrained_model/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/train/german_model.h5' \
# --output_path=$root'train_dnn/train/german_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/german_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/train/german_model/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/german_a_gated_4_0.3_0.2_p-0.5_p0.6.h5' \
# --output_path=$root'train_dnn/gated_models/german_a_gated_4_0.3_0.2_p-0.5_p0.6/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/german_a_gated_4_0.3_0.2_p-0.5_p0.6/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/german_a_gated_4_0.3_0.2_p-0.5_p0.6/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/german_g_gated_4_0.3_0.2_p-0.1_p0.6.h5' \
# --output_path=$root'train_dnn/gated_models/german_g_gated_4_0.3_0.2_p-0.1_p0.6/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/german_g_gated_4_0.3_0.2_p-0.1_p0.6/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/german_g_gated_4_0.3_0.2_p-0.1_p0.6/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=credit \
# --dataset_path=$root'table/credit/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/german_g&a_gated_4_0.3_0.2_p-1.0_p0.25.h5' \
# --output_path=$root'train_dnn/gated_models/german_g&a_gated_4_0.3_0.2_p-1.0_p0.25/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/german_g&a_gated_4_0.3_0.2_p-1.0_p0.25/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/german_g&a_gated_4_0.3_0.2_p-1.0_p0.25/predict.log'   & 



# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/compas_model_g.h5' \
# --output_path=$root'train_dnn/models_flip/compas_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/compas_model_g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/compas_model_g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/compas_model_r.h5' \
# --output_path=$root'train_dnn/models_flip/compas_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/compas_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/compas_model_r/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/compas_model_r&g.h5' \
# --output_path=$root'train_dnn/models_flip/compas_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/compas_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/compas_model_r&g/predict.log'   & 



# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/compas_model_g.h5' \
# --output_path=$root'train_dnn/models_adv/compas_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/compas_model_g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/compas_model_g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/compas_model_r.h5' \
# --output_path=$root'train_dnn/models_adv/compas_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/compas_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/compas_model_r/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/compas_model_r&g.h5' \
# --output_path=$root'train_dnn/models_adv/compas_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/compas_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/compas_model_r&g/predict.log'   & 


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/compas_scores.h5' \
# --output_path=$root'train_dnn/multitask_models/compas_scores/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/compas_scores/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/multitask_models/compas_scores/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/retrained_models/compas_EIDIG_INF_retrained_model/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/train/compas_model.h5' \
# --output_path=$root'train_dnn/train/compas_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/compas_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/train/compas_model/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4.h5' \
# --output_path=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5.h5' \
# --output_path=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45.h5' \
# --output_path=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/predict.log'   & 


# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/lsac_model_g.h5' \
# --output_path=$root'train_dnn/models_adv/lsac_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/lsac_model_g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/lsac_model_g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/lsac_model_r.h5' \
# --output_path=$root'train_dnn/models_adv/lsac_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/lsac_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/lsac_model_r/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_adv/lsac_model_r&g.h5' \
# --output_path=$root'train_dnn/models_adv/lsac_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_adv/lsac_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_adv/lsac_model_r&g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/lsac_model_g.h5' \
# --output_path=$root'train_dnn/models_flip/lsac_model_g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/lsac_model_g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/lsac_model_g/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/lsac_model_r.h5' \
# --output_path=$root'train_dnn/models_flip/lsac_model_r/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/lsac_model_r/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/lsac_model_r/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/models_flip/lsac_model_r&g.h5' \
# --output_path=$root'train_dnn/models_flip/lsac_model_r&g/predict_scores.npy' \
# --output_path2=$root'train_dnn/models_flip/lsac_model_r&g/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/models_flip/lsac_model_r&g/predict.log'   & 


# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/multitask_models/lsac.h5' \
# --output_path=$root'train_dnn/multitask_models/lsac/predict_scores.npy' \
# --output_path2=$root'train_dnn/multitask_models/lsac/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/multitask_models/lsac/predict.log'   & &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model.h5' \
# --output_path=$root'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/train/lsac_model.h5' \
# --output_path=$root'train_dnn/train/lsac_model/predict_scores.npy' \
# --output_path2=$root'train_dnn/train/lsac_model/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/train/lsac_model/predict.log'   & &

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/lsac_g_gated_4_0.3_0.2_p-0.85_p0.2.h5' \
# --output_path=$root'train_dnn/gated_models/lsac_g_gated_4_0.3_0.2_p-0.85_p0.2/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/lsac_g_gated_4_0.3_0.2_p-0.85_p0.2/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/lsac_g_gated_4_0.3_0.2_p-0.85_p0.2/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/lsac_r_gated_4_0.3_0.2_p-0.9_p0.05.h5' \
# --output_path=$root'train_dnn/gated_models/lsac_r_gated_4_0.3_0.2_p-0.9_p0.05/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/lsac_r_gated_4_0.3_0.2_p-0.9_p0.05/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/lsac_r_gated_4_0.3_0.2_p-0.9_p0.05/predict.log'   & 

# CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=lsac \
# --dataset_path=$root'table/lsac/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45.h5' \
# --output_path=$root'train_dnn/gated_models/lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45/predict.log'   & 

