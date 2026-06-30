cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'

# pos_map = { 
#             'r': [10],
#             'g': [9],
#             'g&r': [9, 10]
#             }

### lsac
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/lsac_model_r.h5' \
#  --sens_param=11  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/lsac_model_g.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_flip_lsac_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_flip_lsac_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/lsac_model_r&g.h5' \
#  --sens_param=11_10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_flip_lsac_model_r&g_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_flip_lsac_model_r&g__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/lsac_model_g.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 

# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/lsac_model_r.h5' \
#  --sens_param=11  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 

# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/lsac_model_r&g.h5' \
#  --sens_param=11_10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r&g_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_r&g__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 


# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/lsac_g_gated_4_0.3_0.2_p-0.85_p0.2.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_g_gated_4_0.3_0.2_p-0.85_p0.2_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_g_gated_4_0.3_0.2_p-0.85_p0.2__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 


#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45.h5' \
#  --sens_param=11_10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_g&r_gated_4_0.3_0.2_p-0.55_p0.45__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/lsac_r_gated_4_0.3_0.2_p-0.9_p0.05.h5' \
#  --sens_param=11  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_r_gated_4_0.3_0.2_p-0.9_p0.05_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_gated_models_lsac_r_gated_4_0.3_0.2_p-0.9_p0.05__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 

### lsac locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/lsac_model.h5' \
#  --sens_param=10 --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/lsac_model.h5' \
#  --sens_param=11 --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &


# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/lsac_model.h5' \
#  --sens_param=11_10 --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_train_lsac_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

# ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

# ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model.h5' \
#  --sens_param=11  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

#  ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/lsac_EIDIG_INF_retrained_model.h5' \
#  --sens_param=11_10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_retrained_models_lsac_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &


# #  ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/lsac.h5' \
#  --sens_param=10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

# ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/lsac.h5' \
#  --sens_param=11  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

#  ### lsac locked
# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='lsac_r_g' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/lsac.h5' \
#  --sens_param=11_10  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac_.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_multitask_models_lsac__svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &


CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
 --exp_name='lsac_r_g' \
 --dataset='lsac' \
 --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/models_adv/lsac_model_0.8_r&g.h5' \
 --sens_param=11_10  --max_global=1000000  \
 --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_0.8_r&g_.npy' \
 --svm_file=$baseroot'train_boundaries/lsac/._exp_train_dnn_models_adv_lsac_model_0.8_r&g__svm.npy' \
 --gan_file=$baseroot'gans/lsac/lsac_gan.pth'  --experiment=$experiment --step=$step   & 


# # CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='lsac_age' \
#  --dataset='lsac' \
#  --dataset_path=$baseroot'table/lsac/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/lsac_base' \
#  --sens_param=13  --max_global=1000000  \
#  --latent_file=$baseroot'table/lsac/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/lsac/lsac.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/lsac/lsac_svm.npy' \
#  --gan_file=$baseroot'gans/lsac/lsac_gan.pth' --experiment=$experiment --step=$step   &

# ### bank

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='bank_age' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_base' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/bank/bank.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/bank/bank_svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step   &

# ### meps
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='meps_sex' \
#  --dataset='meps' \
#  --dataset_path=$baseroot'table/meps/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/meps_base' \
#  --sens_param=3  --max_global=1000000  \
#  --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/lsac/meps/meps.npy' \
#  --svm_file=$baseroot'train_boundaries/lsac/meps/meps_svm.npy' \
#  --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step   &