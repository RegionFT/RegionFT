cd ../table_model

root='../exp/'


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

CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=compas \
--dataset_path=$root'table/compas/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4.h5' \
--output_path=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/labels.npy' \
2>&1 | tee '../exp/train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4/predict.log'   & 

# CUDA_VISIBLE_DEVICES=2 python model_predict.py --dataset=compas \
# --dataset_path=$root'table/compas/sampled_data.csv' \
# --model_path=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5.h5' \
# --output_path=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/predict_scores.npy' \
# --output_path2=$root'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/labels.npy' \
# 2>&1 | tee '../exp/train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5/predict.log'   & 

CUDA_VISIBLE_DEVICES=3 python model_predict.py --dataset=compas \
--dataset_path=$root'table/compas/sampled_data.csv' \
--model_path=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45.h5' \
--output_path=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/predict_scores.npy' \
--output_path2=$root'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/labels.npy' \
2>&1 | tee '../exp/train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45/predict.log'   & 

