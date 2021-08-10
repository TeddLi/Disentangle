#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u ../train_exloss.py \
  --task_name 'cluster_exloss' \
  --train_dirs '../../DSTC8_DATA/Task_4/train' \
  --dev_dirs '../../DSTC8_DATA/Task_4/dev/' \
  --test_dirs '../../DSTC8_DATA/Task_4/test' \
  --vocab_file ../../uncased_L-12_H-768_A-12/vocab.txt\
  --bert_config_file ../../uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint ../../uncased_L-12_H-768_A-12/bert_model.ckpt \
  --warmup_proportion 0.1\
  --batch_size 3 \
  --num_epochs 4 \
  --max_turns 50\
  --log_dir ../logs/\
  --model_dir ../ckpt/\
  --adapt_model_dir ../../Adaptation\
  --cluster_loss_weight 0.2\
  --lr 2e-5 > ../logs/train_exloss.log 2>&1 &

