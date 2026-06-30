cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'
### compas
# pos_map = { 
#             'r': [4],
#             'g': [5],
#             'g&r': [5, 4]
#             }


# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='compas_r' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/compas_model_r.h5' \
#  --sens_param=5  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='compas_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/compas_model_g.h5' \
#  --sens_param=6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_flip_compas_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_flip_compas_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='compas_r_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/compas_model_r&g.h5' \
#  --sens_param=5_6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_flip_compas_model_r&g_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_flip_compas_model_r&g__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 


CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
 --exp_name='compas_g' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/compas_model_g.h5' \
 --sens_param=6  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_g_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_g__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 

CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='compas_r' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/compas_model_r&g.h5' \
 --sens_param=5  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r&g_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r&g__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 

CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='compas_r_g' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/compas_model_r&g.h5' \
 --sens_param=5_6  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r&g_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_models_adv_compas_model_r&g__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 


CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='compas_g' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/compas_g_gated_4_0.3_0.2_p-0.1_p0.4.h5' \
 --sens_param=6  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_g_gated_4_0.3_0.2_p-0.1_p0.4_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_g_gated_4_0.3_0.2_p-0.1_p0.4__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 


 CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='compas_r_g' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45.h5' \
 --sens_param=5_6  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_g&r_gated_4_0.3_0.2_p-0.1_p0.45__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 

 CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='compas_r' \
 --dataset='compas' \
 --dataset_path=$baseroot'table/compas/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/gated_models/compas_r_gated_4_0.3_0.2_p-0.1_p0.5.h5' \
 --sens_param=5  --max_global=1000000  \
 --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_r_gated_4_0.3_0.2_p-0.1_p0.5_.npy' \
 --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_gated_models_compas_r_gated_4_0.3_0.2_p-0.1_p0.5__svm.npy' \
 --gan_file=$baseroot'gans/compas/compas_gan.pth'  --experiment=$experiment --step=$step   & 

# ### compas locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='compas_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/compas_model.h5' \
#  --sens_param=6 --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/compas_model.h5' \
#  --sens_param=5 --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &


# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/compas_model.h5' \
#  --sens_param=5_6 --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_train_compas_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# ### compas locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model.h5' \
#  --sens_param=6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# ### compas locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model.h5' \
#  --sens_param=5  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

#  ### compas locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/compas_EIDIG_INF_retrained_model.h5' \
#  --sens_param=5_6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_retrained_models_compas_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# ### compas locked
# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='compas_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/compas_scores.h5' \
#  --sens_param=6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# ### compas locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/compas_scores.h5' \
#  --sens_param=5  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

#  ### compas locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='compas_r_g' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/compas_scores.h5' \
#  --sens_param=5_6  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores_.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/._exp_train_dnn_multitask_models_compas_scores__svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# # CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='compas_age' \
#  --dataset='compas' \
#  --dataset_path=$baseroot'table/compas/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/compas_base' \
#  --sens_param=13  --max_global=1000000  \
#  --latent_file=$baseroot'table/compas/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/compas/compas.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/compas/compas_svm.npy' \
#  --gan_file=$baseroot'gans/compas/compas_gan.pth' --experiment=$experiment --step=$step   &

# ### bank

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='bank_age' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_base' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/bank/bank.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/bank/bank_svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step   &

# ### meps
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='meps_sex' \
#  --dataset='meps' \
#  --dataset_path=$baseroot'table/meps/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/meps_base' \
#  --sens_param=3  --max_global=1000000  \
#  --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/compas/meps/meps.npy' \
#  --svm_file=$baseroot'train_boundaries/compas/meps/meps_svm.npy' \
#  --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step   &