cd ../
# CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=census \
#     --num_samples=100 --save='./exp/gans/census/census_gan.pth' \
#     --output='census_sample.csv' \
#     --output_train='census_train_raw.csv' \
#     --data='./my_census_train.csv'

CUDA_VISIBLE_DEVICES=3 python train_gan.py --exp_name=credit \
    --num_samples=100 --save='./exp/gans/credit/credit_gan.pth' \
    --output='credit_sample.csv' \
    --output_train='credit_train_raw.csv' \
    --data='./my_credit_train.csv'

CUDA_VISIBLE_DEVICES=2 python train_gan.py --exp_name=bank \
    --num_samples=100 --save='./exp/gans/bank/bank_gan.pth' \
    --output='bank_sample.csv' \
    --output_train='bank_train_raw.csv' \
    --data='./my_bank_train.csv'

CUDA_VISIBLE_DEVICES=1 python train_gan.py --exp_name=compas \
    --num_samples=100 --save='./exp/gans/compas/compas_gan.pth' \
    --output='compas_sample.csv' \
    --output_train='compas_train_raw.csv' \
    --data='./my_compas_train.csv'
CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=lsac \
    --num_samples=100 --save='./exp/gans/lsac/lsac_gan.pth' \
    --output='lsac_sample.csv' \
    --output_train='lsac_train_raw1w.csv' \
    --data='./my_lsac_train.csv'


# CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=meps \
#     --num_samples=100 --save='meps_gan.pth' \
#     --output='meps_sample.csv' \
#     --output_train='meps_train_raw.csv' \
#     --data='./datasets/meps_train.csv'    

## the gans are stored in ../exp/gans manually