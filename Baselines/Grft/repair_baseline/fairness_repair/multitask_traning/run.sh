
CUDA_VISIBLE_DEVICES=0 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train_bankdataset.py\
 --config_str="0-60|true|1|-1|-1|-1;"  --dataset pre_bank_marketing  \
 2>&1 >>b.log &





CUDA_VISIBLE_DEVICES=1 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train_bankdataset.py\
 --config_str="0-60|true|1|1|1|1;"   --dataset pre_bank_marketing  \
 2>&1 >>b.log &






CUDA_VISIBLE_DEVICES=2 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train_bankdataset.py\
 --config_str="0-60|true|10|-1|-1|-1;"    --dataset pre_bank_marketing  \
 2>&1 >>b.log &




 
CUDA_VISIBLE_DEVICES=3 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train_bankdataset.py\
 --config_str="0-60|true|10|-1|-0.1|-0.1;"   --dataset pre_bank_marketing   \
 2>&1 >>b.log &








 
 


CUDA_VISIBLE_DEVICES=0 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train.py\
 --config_str="0-60|true|1|-1|-1|-1;"   \
 2>&1 >>a.log &





CUDA_VISIBLE_DEVICES=1 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train.py\
 --config_str="0-60|true|1|1|1|1;"    \
 2>&1 >>a.log &






CUDA_VISIBLE_DEVICES=2 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train.py\
 --config_str="0-60|true|10|-1|-1|-1;"     \
 2>&1 >>a.log &




 
CUDA_VISIBLE_DEVICES=3 \
nohup /home/malei/anaconda3/envs/tf13/bin/python multitask_train.py\
 --config_str="0-60|true|10|-1|-0.1|-0.1;"     \
 2>&1 >>a.log &

