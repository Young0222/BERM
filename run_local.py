#!/usr/bin/env python
#coding=utf-8

#import argparse
import shutil
import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
import random
import numpy as np
import tensorflow as tf

from char_model_fn_post import model_fn
#from char_model_fn import model_fn
import char_input_fn as input_fn
import checkpoint_util.checkpoint_utils as ckpt_utils
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
# specialized distributed
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-local, 1-single_dist, 2-multi_dist,9n}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# common
tf.app.flags.DEFINE_string("profile_dir", "", "model profile paths")
tf.app.flags.DEFINE_integer("log_steps", 100, "save summary every steps")
tf.app.flags.DEFINE_string("model_dir", "./model_170m_reload/", "model check point dir")
tf.app.flags.DEFINE_string("checkpoint_to_export", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", "./model.bak/", "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
tf.flags.DEFINE_string("best_metric_to_export", "auc:up:2", "bestmetric to export")   ## auc:up:2,loss:down:2
tf.flags.DEFINE_string("log_dir", "nohup.out_reload", "log dir")

#################### model Arguments ##################
model_params = {
        "gpus_list": "3"  ## many gpus indicates tower grads update
    }

#################### DATA Arguments ####################


def set_dist_env():
    if FLAGS.dist_mode == 0:   # local
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if len(model_params["gpus_list"])==0 or FLAGS.task_type != "train" else model_params["gpus_list"]
    elif FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        if job_name == "ps" or job_name == 'evaluator':
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        elif args.role == "chief":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task_index)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task_index + 1)
        # local worker参数
        # python3.6 run_local.py --role=ps --index=0
        # python3.6 run_local.py --role=chief --index=0
        # python3.6 run_local.py --role=worker --index=0
        # python3.6 run_local.py --role=evaluato/r --index=0
        cluster = {'chief':['localhost:2220'], 'worker':['localhost:2223'], 'evaluator':['localhost:2221'], 'ps': ['localhost:2222']}
        tf_config = {
            'cluster': cluster,
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        #获取cluster内容
        cluster_def = eval(os.getenv('TF_CONFIG'))['cluster']
        print(json.dumps(cluster_def))
        cluster = tf.train.ClusterSpec(cluster_def)
        #获取job_name、task_index
        task_def = eval(os.getenv('TF_CONFIG'))['task']
        job_name = task_def['type']
        task_index = task_def['index']
        
        if job_name == "evaluator" and 'evaluator' not in cluster_def:
            def add_evaluator_ips(role_contrast, task_index, cluster_def):
                evaluator_role = []
                ip_str, port = cluster_def[role_contrast].split(':')
                elems = ip_str.split('-')
                elems[2] = 'evaluator'
                for i in range(task_index+1):
                    elems[3] = str(i)
                    evaluator_role.append('-'.join(elems)+':'+port)
                return evaluator_role
            evaluator_role = []
            if "chief" in cluster_def:
                evaluator_role = add_evaluator_ips("chief", task_index, cluster_def)
            elif "worker" in cluster_def:
                evaluator_role = add_evaluator_ips("worker", task_index, cluster_def)
            if len(evaluator_role) > 0:
                cluster_def['evaluator'] = evaluator_role
                    
        tf_config = {
            'cluster': cluster_def,
            'task': {'type': job_name, 'index': task_index }
        }
        print("after revised tf_config")
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    #------check Arguments------
    logger.info("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("")

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            logger.error(e, "at clear_existing_model")
    else:
        logger.info("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    #------bulid Tasks------
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    if FLAGS.dist_mode != 0:
        dist_strategy = tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
        config = tf.estimator.RunConfig(
                 train_distribute=dist_strategy,
                 eval_distribute=dist_strategy,
                 log_step_count_steps = FLAGS.log_steps,
                 save_summary_steps = FLAGS.log_steps,
                 session_config = session_config,
                 save_checkpoints_steps = 200)
    else:
        config = tf.estimator.RunConfig().replace(
                 log_step_count_steps = FLAGS.log_steps,
                 save_summary_steps = FLAGS.log_steps,
                 session_config = session_config,
                 save_checkpoints_steps = FLAGS.log_steps,
                 keep_checkpoint_max = 0)
    DeepModel = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn.train_input_fn())#, hooks=[logging_hook])
        eval_hooks = []
        eval_hooks.append(ckpt_utils.CkptDeletionHook(FLAGS.log_dir, FLAGS.best_metric_to_export, FLAGS.model_dir, "evaluate" ))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn.eval_input_fn(), steps=None, hooks=eval_hooks, start_delay_secs=30, throttle_secs=20)
        if len(FLAGS.profile_dir) > 1:
            with tf.contrib.tfprof.ProfileContext(FLAGS.profile_dir) as pctx:
                tf.estimator.train_and_evaluate(DeepModel, train_spec, eval_spec)
        else:
            tf.estimator.train_and_evaluate(DeepModel, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        DeepModel.evaluate(input_fn=lambda: input_fn.eval_input_fn())
    elif FLAGS.task_type == 'infer':
        preds = DeepModel.predict(input_fn=lambda: input_fn.predict_input_fn(), predict_keys="probabilities")
        for prob in preds:
            print(prob['probabilities'])
    elif FLAGS.task_type == 'export':
        def export_best_ckpt(feature_spec, model_path, checkpoint_to_export, predictor):
            import mix_utils as ckpt_utils
            best_ckpt = checkpoint_to_export
            servable_model_path, ckpt_path = ckpt_utils.export_model_graph(
                                             feature_spec, predictor, model_dir=model_path, ckpt=best_ckpt)

        export_best_ckpt(feature_spec = input_fn.export_input_fn(),
                          model_path=FLAGS.model_dir,
                          checkpoint_to_export=FLAGS.checkpoint_to_export,
                          predictor=DeepModel
         )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
