# for ((i=1;i<21;i++))
# do
#   for ((j=1;j<20;j++))
#   do
#       CUDA_VISIBLE_DEVICES=$[j%7+1] nohup python repair/repair_german.py --attr g\&a --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
# #      CUDA_VISIBLE_DEVICES=$[j-1] nohup  python repair_german.py --attr g --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
# #      CUDA_VISIBLE_DEVICES=$[j%8] nohup  python repair_german.py --attr g\&a --p0 $[i] --p1 $[j] --acc_lb 0.7 --percent 0.3 --weight_threshold 0.2 2>&1 >german.log &
#   done
#   sleep 2m
#   wait
#   echo -n "$[i] is finished!";
# done

array=("r" "g" "r\&g")
per=(0.1,0.2,0.3,0.4,0.5)
wei=(0.1,0.2,0.3,0.4,0.5)

for a in ${array[@]}
do 
    echo $a a
done