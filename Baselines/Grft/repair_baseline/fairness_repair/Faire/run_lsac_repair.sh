# for ((i=1;i<21;i++))
# do
#   for ((j=1;j<20;j++))
#   do
# #    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr a\&g --p0 $[i] --p1 $[j] --acc_lb 0.825 2>&1 >adult_a\&g.log  &
#      CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair/repair_lsac.py --attr g\&r --p0 $[i] --p1 $[j] --acc_lb 0.5 --percent 0.3 --weight_threshold 0.2 2>&1 >backup/lsac_a\&r.log &
#   done
#   sleep 2m
#   wait
#   echo -n "$[i] is finished!";
# done
array=("r" "g" "r\&g")
per=(0.1 0.2 0.3 0.4 0.5)
wei=(0.1 0.2 0.3 0.4 0.5)
for p in ${per[@]}
do
    for w in ${wei[@]}
    do
        for a in ${array[@]}
        do
            for ((j=1;j<4;j++))
            do
                CUDA_VISIBLE_DEVICES=$[j%7 + 1] nohup python repair/repair_lsac_ablation.py --attr $a --ablation $j --percent $p --weight_threshold $w 2>&1 >adult_a\&r.log &
            done
            sleep 1m
            wait
            echo -n "$attr is finished!";
        done
        sleep 5m
        wait
    done
    sleep 5m
    wait
done
