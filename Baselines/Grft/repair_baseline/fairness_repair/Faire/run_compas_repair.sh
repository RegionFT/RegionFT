for ((i=11;i<21;i++))
do
  for ((j=1;j<10;j++))
  do
#    CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_adult.py --attr a\&g --p0 $[i] --p1 $[j] --acc_lb 0.825 2>&1 >adult_a\&g.log  &
     CUDA_VISIBLE_DEVICES=$[j%8] nohup python repair_compas.py --attr g\&r --p0 $[i] --p1 $[j] --acc_lb 0.5 --percent 0.2 --weight_threshold 0.2 2>&1 >compas_a\&r.log &
  done
  sleep 2m
  wait
  echo -n "$[i] is finished!";
done

