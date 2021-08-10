#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

set -e

i=0
for((i;i<4;i=i+1))
do
    str1="epoch"${i}
    str2="epoch_"${i}"_eval_results.txt"
    str3="epoch_"${i}"_test_results.txt"

    echo $str1 >> result.log
    echo $str2 >> result.log
    python3 ../../../task-4-evaluation.py --gold ../../../DSTC8_DATA/Task_4/dev/*anno* --auto $str2 >> result.log
    echo $str3 >> result.log
    python3 ../../../task-4-evaluation.py --gold ../../../DSTC8_DATA/Task_4/test/*anno* --auto $str3 >> result.log
done

