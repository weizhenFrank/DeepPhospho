gpu_idx=$1

cat models_params.txt | while IFS=":" read i j
do
if [ $i -gt 3 ] && [ $i -lt 9 ]
then
  if [ `expr $gpu_idx % $2` -gt 99 ]
  then
    run_gpu_idx=`expr $(expr $gpu_idx % $2) + 1`
  else
    run_gpu_idx=`expr $(expr $gpu_idx % $2) + 0`
  fi

if [ $gpu_idx -eq $1 ]
then
  python main.py --pretrain_param $j --GPU $run_gpu_idx --ad_hoc $i --exp_name $i"_"$3  &
else
  tmux split-window -h "python main.py --pretrain_param $j --GPU $run_gpu_idx --ad_hoc $i --exp_name $i"_"$3  & ; read"
fi

#  echo "python main.py --pretrain_param $j --GPU $run_gpu_idx --ad_hoc $i --exp_name $i"_"$3 "

  gpu_idx=$(expr $gpu_idx + 1)
else
  continue
fi
done
tmux select-layout tiled