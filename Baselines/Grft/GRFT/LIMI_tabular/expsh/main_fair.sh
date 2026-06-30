cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'

# pos_map = { 'a': [0],
#             'r': [6],
#             'g': [7],
#             'a&r': [0, 6],
#             'a&g': [0, 7],
#             'r&g': [6, 7]
#             }



### census
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_adv/adult_model_a.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_r_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_r&g.h5' \
#  --sens_param=7_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_r&g_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_r&g__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_r.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_r_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_r__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_g.h5' \
#  --sens_param=8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_g_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_g__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_a_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_a&r.h5' \
#  --sens_param=1_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a&r_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a&r__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_a_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_a&g.h5' \
#  --sens_param=1_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a&g_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a&g__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/models_flip/adult_model_a.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_flip_adult_model_a__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# # CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
# #  --exp_name='census_g' \
# #  --dataset='census' \
# #  --dataset_path=$baseroot'table/census/sampled_data.csv' \
# #  --model_path=$baseroot'train_dnn/models_adv/adult_model_g.h5' \
# #  --sens_param=8  --max_global=1000000  \
# #  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
# #  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_g_.npy' \
# #  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_g__svm.npy' \
# #  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# # CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
# #  --exp_name='census_r' \
# #  --dataset='census' \
# #  --dataset_path=$baseroot'table/census/sampled_data.csv' \
# #  --model_path=$baseroot'train_dnn/models_adv/adult_model_r.h5' \
# #  --sens_param=7  --max_global=1000000  \
# #  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
# #  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_r_.npy' \
# #  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_r__svm.npy' \
# #  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# # CUDA_VISIBLE_DEVICES=3 python main_fair_ours.py \
# #  --exp_name='census_a_r' \
# #  --dataset='census' \
# #  --dataset_path=$baseroot'table/census/sampled_data.csv' \
# #  --model_path=$baseroot'train_dnn/models_adv/adult_model_a&r.h5' \
# #  --sens_param=1_7  --max_global=1000000  \
# #  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
# #  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a&r_.npy' \
# #  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a&r__svm.npy' \
# #  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# # CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
# #  --exp_name='census_a_g' \
# #  --dataset='census' \
# #  --dataset_path=$baseroot'table/census/sampled_data.csv' \
# #  --model_path=$baseroot'train_dnn/models_adv/adult_model_a&g.h5' \
# #  --sens_param=1_8  --max_global=1000000  \
# #  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
# #  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a&g_.npy' \
# #  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_a&g__svm.npy' \
# #  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &


# # CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
# #  --exp_name='census_r_g' \
# #  --dataset='census' \
# #  --dataset_path=$baseroot'table/census/sampled_data.csv' \
# #  --model_path=$baseroot'train_dnn/models_adv/adult_model_r&g.h5' \
# #  --sens_param=7_8  --max_global=1000000  \
# #  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
# #  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_r&g_.npy' \
# #  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_models_adv_adult_model_r&g__svm.npy' \
# #  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &



# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_r_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8.h5' \
#  --sens_param=7_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_r&g_gated_4_0.3_0.2_p-0.9_p0.8__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_r_gated_4_0.3_0.2_p-0.95_p0.8.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_r_gated_4_0.3_0.2_p-0.95_p0.8_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_r_gated_4_0.3_0.2_p-0.95_p0.8__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_g_gated_4_0.3_0.2_p-0.6_p0.1.h5' \
#  --sens_param=8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_g_gated_4_0.3_0.2_p-0.6_p0.1_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_g_gated_4_0.3_0.2_p-0.6_p0.1__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &


#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_a_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25.h5' \
#  --sens_param=1_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_a_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_a&r_gated_4_0.3_0.2_p-0.35_p0.25.h5' \
#  --sens_param=1_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a&g_gated_4_0.3_0.2_p-0.3_p0.25__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/gated_models/adult_a_gated_4_0.3_0.2_p-0.3_p0.15.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a_gated_4_0.3_0.2_p-0.3_p0.15_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_gated_models_adult_a_gated_4_0.3_0.2_p-0.3_p0.15__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step &




# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
#  --sens_param=8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='census_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

 CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
 --exp_name='census_a_g' \
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
 --sens_param=1_8  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
 --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
#  --sens_param=1_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_r&g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/retrained_models/adult_EIDIG_INF_retrained_model.h5' \
#  --sens_param=7_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_retrained_models_adult_EIDIG_INF_retrained_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &


# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='census_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='census_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=1_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_a_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=1_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_r&g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/multitask_models/adult_model.h5' \
#  --sens_param=7_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_multitask_models_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &



# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

# CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='census_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=1 python main_fair_ours.py \
#  --exp_name='census_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='census_a_g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=1_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_a_r' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=1_7  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &

#  CUDA_VISIBLE_DEVICES=2 python main_fair_ours.py \
#  --exp_name='census_r&g' \
#  --dataset='census' \
#  --dataset_path=$baseroot'table/census/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/adult_model.h5' \
#  --sens_param=7_8  --max_global=1000000  \
#  --latent_file=$baseroot'table/census/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model_.npy' \
#  --svm_file=$baseroot'train_boundaries/census/._exp_train_dnn_train_adult_model__svm.npy' \
#  --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step &






# ### credit locked
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_gender' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/credit_base' \
#  --sens_param=9  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/credit/credit.npy' \
#  --svm_file=$baseroot'train_boundaries/census/credit/credit_svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# # CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='credit_age' \
#  --dataset='credit' \
#  --dataset_path=$baseroot'table/credit/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/credit_base' \
#  --sens_param=13  --max_global=1000000  \
#  --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/credit/credit.npy' \
#  --svm_file=$baseroot'train_boundaries/census/credit/credit_svm.npy' \
#  --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

# ### bank

# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='bank_age' \
#  --dataset='bank' \
#  --dataset_path=$baseroot'table/bank/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/bank_base' \
#  --sens_param=1  --max_global=1000000  \
#  --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/bank/bank.npy' \
#  --svm_file=$baseroot'train_boundaries/census/bank/bank_svm.npy' \
#  --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step

# ### meps
# CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
#  --exp_name='meps_sex' \
#  --dataset='meps' \
#  --dataset_path=$baseroot'table/meps/sampled_data.csv' \
#  --model_path=$baseroot'train_dnn/train/meps_base' \
#  --sens_param=3  --max_global=1000000  \
#  --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
#  --boundary_file=$baseroot'train_boundaries/census/meps/meps.npy' \
#  --svm_file=$baseroot'train_boundaries/census/meps/meps_svm.npy' \
#  --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step