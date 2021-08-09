# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import logging
import datetime
import time
import tensorflow as tf
import modeling as modeling
import operator, os, tokenization
import time, random, copy, json
from data_help import *
from Model_exloss import *
from config import config
import pdb
import optimization as optimization
from tensorflow.python import pywrap_tensorflow

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def assign(variable, value, sess):
    assign_op = tf.assign(variable, value)
    sess.run(assign_op)


def exloss_weight(cl_list, weight):
    result = []
    for i in range(len(cl_list)):
        anchor = cl_list[i][-1]
        temp = [weight*((index)/ 50) if x == anchor else 0 for index,x in enumerate(cl_list[i])]
        temp[-1] = weight*0.8
        result.append(temp)
    return result


# get the variable collections (a list, which are then used to assign the value)
def get_variables_via_scope(scope_list):
    variables = []
    for scope in scope_list:
        scope_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        variables += scope_variable
    return variables



def evaluate_final(epoch, sess, model):
    # ------------------------- define parameter -----------------------------
    dev_dirs = config.test_dirs
    bert_config_file = config.bert_config_file
    vocab_file = config.vocab_file
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    model_dir = config.model_dir
    taskname = config.task_name
    model_dir = os.path.join(model_dir, taskname)
    max_turns = config.max_turns

    dev_paths = os.listdir(dev_dirs)
    dev_paths = [os.path.join(dev_dirs, path) for path in dev_paths if path.endswith('.annotation.txt')]
    dev_instances, cluster_dic = read_data(dev_paths, is_test=True)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    print('load data')
    dev_dataset, data_query, data_name = load_data(dev_instances, max_turns)
    print('compute cluster')
    data_cluster = compute_cluster(data_query, data_name, cluster_dic)
    print('convert index')
    dataset_cluster = convert_index(data_cluster)
    print('data Done')
    num_turns = 50
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def test_run_step(sess, dial_input_ids, dial_input_masks, dial_token_types, dial_masks, target_index, candi_indexs,
                      batch_features, data_cluster, model):
        feed_dict = {model.dial_input_ids: dial_input_ids,
                     model.dial_input_masks: dial_input_masks,
                     model.dial_token_types: dial_token_types,
                     model.dial_masks: dial_masks,
                     # model.dial_features: batch_features,
                     model.target_index: target_index,
                     model.candi_indexs: candi_indexs,
                     model.is_training: False
                     }
        probs, preds = sess.run([model.probabilities, model.preds], feed_dict)
        return probs, preds


    f = open(os.path.join(model_dir, 'epoch_{}_test_results.txt'.format(epoch)), 'w')
    f1 = open(os.path.join(model_dir, 'epoch_{}_test_detail.txt'.format(epoch)), 'w')
    for i in range(0, len(dev_dataset), batch_size):
        ori_batch = dev_instances[i:i + batch_size]
        dev_batch = dev_dataset[i:i + batch_size]
        data_cluster = dataset_cluster[i:i + batch_size]
        # pdb.set_trace()
        batch_input_ids, batch_input_masks, batch_token_types, batch_masks, batch_labels, batch_target_index, batch_candi_indexs, batch_features, batch_gold, data_cluster = batch2idxs(
            dev_batch, data_cluster, max_turns, tokenizer, max_seq_length)
        probs, preds = test_run_step(sess, batch_input_ids, batch_input_masks, batch_token_types, batch_masks, batch_target_index, batch_candi_indexs, batch_features, data_cluster, model)
        for j in range(len(ori_batch)):
            name = ori_batch[j][0]
            query = ori_batch[j][1]
            pred = preds[j]
            probs_num = probs[j]
            pred_link = query - (max_turns - 1 - pred)
            # print("{}:{} {} -".format(name, query, pred_link))
            f.write(name + ':' + str(query) + ' ' + str(pred_link) + ' ' + '-' + '\n')
            f1.write(name + ':' + str(query) + ' ' + str(pred_link) + ' ' + str(probs_num) + '\n')
    f.close()
    f1.close()


def evaluate(epoch, sess, model):
    # ------------------------- define parameter -----------------------------
    dev_dirs = config.dev_dirs
    bert_config_file = config.bert_config_file
    vocab_file = config.vocab_file
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    model_dir = config.model_dir
    taskname = config.task_name
    model_dir = os.path.join(model_dir, taskname)
    max_turns = config.max_turns

    dev_paths = os.listdir(dev_dirs)
    dev_paths = [os.path.join(dev_dirs, path) for path in dev_paths if path.endswith('.annotation.txt')]
    dev_instances, cluster_dic = read_data(dev_paths, is_test=True)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    print('load data')
    dev_dataset, data_query, data_name = load_data(dev_instances, max_turns)
    print('compute cluster')
    data_cluster = compute_cluster(data_query, data_name, cluster_dic)
    print('convert index')
    dataset_cluster = convert_index(data_cluster)
    print('data Done')
    num_turns = 50
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def eval_run_step(sess, dial_input_ids, dial_input_masks, dial_token_types, dial_masks, target_index, candi_indexs,
                      batch_features, data_cluster, model):
        feed_dict = {model.dial_input_ids: dial_input_ids,
                     model.dial_input_masks: dial_input_masks,
                     model.dial_token_types: dial_token_types,
                     model.dial_masks: dial_masks,
                     # model.dial_features: batch_features,
                     model.target_index: target_index,
                     model.candi_indexs: candi_indexs,
                     model.is_training: False
                     }
        probs, preds = sess.run([model.probabilities, model.preds], feed_dict)

        return probs, preds

    f = open(os.path.join(model_dir, 'epoch_{}_eval_results.txt'.format(epoch)), 'w')
    f1 = open(os.path.join(model_dir, 'epoch_{}_eval_detail.txt'.format(epoch)), 'w')
    match = 0
    total = 0
    for i in range(0, len(dev_dataset), batch_size):
        ori_batch = dev_instances[i:i + batch_size]
        dev_batch = dev_dataset[i:i + batch_size]
        data_cluster = dataset_cluster[i:i + batch_size]
        # pdb.set_trace()
        batch_input_ids, batch_input_masks, batch_token_types, batch_masks, batch_labels, batch_target_index, batch_candi_indexs, batch_features, batch_gold, data_cluster = batch2idxs(
            dev_batch, data_cluster, max_turns, tokenizer, max_seq_length)
        probs, preds = eval_run_step(sess, batch_input_ids, batch_input_masks, batch_token_types, batch_masks, batch_target_index, batch_candi_indexs,batch_features, data_cluster, model)


        for i in range(batch_input_ids.shape[0]):
            if int(preds[i]) in batch_gold[i]:
                match += 1
        total += batch_input_ids.shape[0]

        if i % 100 == 0:
            print("total {}, match {}, acc {} -".format(total, match, match/total))
        for j in range(len(ori_batch)):
            name = ori_batch[j][0]
            query = ori_batch[j][1]
            pred = preds[j]
            probs_num = probs[j]
            pred_link = query - (max_turns - 1 - pred)
            # print("{}:{} {} -".format(name, query, pred_link))
            f.write(name + ':' + str(query) + ' ' + str(pred_link) + ' ' + '-' + '\n')
            f1.write(name + ':' + str(query) + ' ' + str(pred_link) + ' ' + str(probs_num) + '\n')
    f.close()
    f1.close()









# Start training
def train():
    # ------------------------- define parameter -----------------------------
    train_dirs = config.train_dirs
    bert_config_file = config.bert_config_file
    init_checkpoint = config.init_checkpoint
    vocab_file = config.vocab_file
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    warmup_proportion = config.warmup_proportion
    log_dir = config.log_dir
    model_dir = config.model_dir
    lr = config.lr
    max_turns = config.max_turns
    task_name = config.task_name
    adapt_model_dir = config.adapt_model_dir
    cluster_loss_weight = config.cluster_loss_weight


    train_paths = os.listdir(train_dirs)
    train_paths = [os.path.join(train_dirs, path) for path in train_paths if path.endswith('.annotation.txt')]
    train_instances, cluster_dic = read_data(train_paths, is_test=False)
    num_train_steps = int(len(train_instances) / batch_size * num_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    print('load data')
    train_dataset, data_query, data_name = load_data(train_instances, max_turns)
    print('compute cluster')
    data_cluster = compute_cluster(data_query, data_name, cluster_dic)
    print('convert index')
    data_cluster = convert_index(data_cluster)
    num_turns = 50
    print('data Done')

    def train_run_step(sess, dial_input_ids, dial_input_masks, dial_token_types, dial_masks, labels, target_index,
                       candi_indexs, batch_features, dial_golds, data_cluster, model):
        start_time = time.time()
        feed_dict = {model.dial_input_ids: dial_input_ids,
                     model.dial_input_masks: dial_input_masks,
                     model.dial_token_types: dial_token_types,
                     model.dial_masks: dial_masks,
                     model.labels: labels,
                     # model.dial_features: batch_features,
                     model.target_index: target_index,
                     model.candi_indexs: candi_indexs,
                     model.data_cluster: data_cluster,
                     model.is_training: True
                     }
        # pdb.set_trace()
        _, step, probs, loss, preds, debug_package_ = sess.run([train_op, global_step, model.probabilities, model.loss, model.preds, model.debug_package],
                                               feed_dict)

        # after_sort, sort_array, temp, temp1, temp2, temp3, temp4, valid_index, new_index, cluster_loss = debug_package_

        match = 0
        for i in range(dial_input_ids.shape[0]):
            if int(preds[i]) in dial_golds[i]:
                match += 1
        match_acc = match / dial_input_ids.shape[0]
        time_str = datetime.datetime.now().isoformat()
        time_elapsed = time.time() - start_time
        logger.info(
            "%s: step %s, loss %s, match_acc %s, %6.7f secs/batch" % (time_str, step, loss, match_acc, time_elapsed))
        return loss

    # ------------------------------------------make some dirs-------------------------------------
    if not os.path.exists(log_dir):  # make the log dir to keep logs
        os.makedirs(log_dir)
    else:
        pass

    if not os.path.exists(model_dir):  # make the dir to store the model
        os.makedirs(model_dir)
    else:
        pass

    save_dir = os.path.join(model_dir,task_name)
    if not os.path.exists(save_dir):  # make the log dir to keep logs
        os.makedirs(save_dir)
    else:
        pass

    config_info = json.dumps(config.__dict__, sort_keys=True, indent=4)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:  # write the corresponding config infos
        f.write(config_info)

    # ------------------------------------------check and load the sources--------------------------
    random_seed = 12345
    rng = random.Random(random_seed)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    # ----------------------------- define a logger -------------------------------
    logger = logging.getLogger("execute")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_dir + task_name+"_log.tf", mode="w")
    fh.setLevel(logging.INFO)

    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    fh.setFormatter(formatter)
    logger.addHandler(fh)
    print('Start training...')
    # ----------------------------------- begin to train -----------------------------------
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
            with tf.Session(config=session_conf).as_default() as sess:
                model = DialModel(bert_config, use_one_hot_embeddings=False, num_turns=num_turns)
                model.build_graph()
                global_step = tf.train.get_or_create_global_step()
                tvars = tf.trainable_variables()
                train_op = optimization.create_optimizer(model.loss, lr, num_train_steps, num_warmup_steps,
                                                         use_tpu=False)

                sess.run(tf.initialize_all_variables())

                if init_checkpoint:
                    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                               init_checkpoint)
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                        init_string)
                ckpt = tf.train.get_checkpoint_state(adapt_model_dir)  #
                if ckpt:
                    variables = tf.trainable_variables()
                    variables_to_resotre = [v for v in variables if v.name.split('/')[0] == 'sent_encoder']
                    # variables_to_resotre = [v for v in variables_to_resotre if v.name.split('/')[1] != 'pooler']
                    tf.logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                    saver = tf.train.Saver(variables_to_resotre)
                    saver.restore(sess, ckpt.model_checkpoint_path)

                saver = tf.train.Saver(max_to_keep=20)
                random.seed(123)
                dataset = list(zip(train_dataset, data_cluster))
                for epoch in range(num_epochs):
                    random.shuffle(dataset)
                    train_dataset_, dataset_cluster_ = zip(*dataset)
#                    for i in range(0, len(train_dataset_), batch_size):
#                        train_batch = train_dataset_[i:i + batch_size]
#                        cluster_batch = dataset_cluster_[i:i+batch_size]
#                        cluster_batch = exloss_weight(cluster_batch, cluster_loss_weight)
                        # pdb.set_trace()
#                        batch_input_ids, batch_input_masks, batch_token_types, batch_masks, batch_labels, batch_target_index, batch_candi_indexs, batch_features, batch_gold, data_cluster = batch2idxs(
#                            train_batch, cluster_batch, max_turns, tokenizer, max_seq_length)
                        # pdb.set_trace()
#                        loss = train_run_step(sess, batch_input_ids, batch_input_masks, batch_token_types, batch_masks,
#                                              batch_labels,
#                                              batch_target_index, batch_candi_indexs, batch_features, batch_gold, cluster_batch, model)
                    saver.save(sess, save_dir + '/epoch_' + str(epoch) + '_model.ckpt', epoch)
                    evaluate(epoch, sess, model)
                    evaluate_final(epoch, sess, model)
if __name__ == '__main__':
    print(config)
    train()


