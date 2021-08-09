# -*- coding: utf-8 -*-
import argparse

argparser = argparse.ArgumentParser(description=("Run bert for task 4."))



argparser.add_argument('--task_name', type=str,
                       help=('The dirs of training data'),
                       default = 'baseline')
argparser.add_argument('--train_dirs', type=str,
                       help=('The dirs of training data'),
                       default = '../DSTC8_DATA/Task_4/train_temp')
argparser.add_argument('--dev_dirs', type=str,
                       help=('The dirs of training data'),
                       default = '../DSTC8_DATA/Task_4/dev/')
argparser.add_argument('--test_dirs', type=str,
                       help=('The dirs of training data'),
                       default = '../DSTC8_DATA/Task_4/test/')
argparser.add_argument('--vocab_file', type=str,
                       help=('The path of shared_char_vocab file'),
                       default = '../uncased_L-12_H-768_A-12/vocab.txt')
argparser.add_argument('--bert_config_file', type=str,
                       help=('The path of shared_char_vocab file'),
                       default = '../uncased_L-12_H-768_A-12/bert_config.json')
argparser.add_argument('--init_checkpoint', type=str,
                       help=('The pretrained model'),
                       default = '../uncased_L-12_H-768_A-12/bert_model.ckpt')
argparser.add_argument("--warmup_proportion", type=float, default=0.1,
                       help=("Proportion of training to perform linear learning rate warmup for."))
argparser.add_argument("--batch_size", type=int, default=2,
                       help="Number of instances per batch.")
argparser.add_argument("--batch_multiplier", type=int, default=1,
                       help="Number of multiplier per batch.")
argparser.add_argument("--num_epochs", type=int, default=10,
                       help=("Number of epochs to perform in training."))
argparser.add_argument("--max_seq_length", type=int, default=100,
                       help=("The maximum length of a sentence at the char level. Longer sentences will be truncated, and shorter ones will be padded."))
argparser.add_argument("--max_turns", type=int, default=50,
                       help=("The maximum turn of the dialogue"))
argparser.add_argument("--log_dir", type=str, default='./logs/',
                       help=("Directory to save logs to."))
argparser.add_argument("--model_dir", type=str, default='./ckpt/',
                       help=("Directory to save model checkpoints to."))
argparser.add_argument("--adapt_model_dir", type=str, default='../Adaptation',
                       help=("Directory to save model checkpoints to."))
argparser.add_argument("--lr", type=float, default=2e-5,#5e-5,#0.0001,
                       help=('the initial learning rate'))
argparser.add_argument("--cluster_loss_weight", type=float, default=0.1,
                       help=('cluster_loss_weight'))
argparser.add_argument("--tree_loss_weight", type=float, default=0.5,#5e-5,#0.0001,
                       help=('cluster_loss_weight'))
config = argparser.parse_args()
