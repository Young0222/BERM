#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import random
import math
import numpy as np
import char_input_fn


end_points = {}

class ModelConfig(object):
    def __init__(self):
        # ------hyperparameters----
        self.query_size = 10 #config['query_size']
        self.title_size = 65 #config['title_size']
        self.vocab_size = 40000 #config['vocab_size']
        self.embedding_dim = 128 #config['embedding_dim']
        self.text_cnn_filter = [1] #[int(ele) for ele in config["text_cnn_filter"].split(',')]
        self.deep_layers = [1024,256,64] #[int(ele) for ele in config["deep_layers"].split(',')] if len(config["deep_layers"]) > 0 else []
        self.learning_rate = 0.001 #config['learning_rate']
        self.l2_reg = 0.000001 #config["l2_reg"]
        self.keep_prob = 0.9 #config['keep_prob']
        self.vocab_file = "./word2vecmodel.txt"
        self.gpus_list = [] #[int(ele) for ele in config['gpus_list'].split(",")] if len(config['gpus_list']) > 1 else []
        self.optimizer = None

# global vars
model_config = ModelConfig()


def word2vec_initiliar(filename, vocab_size, embedded_dim, value=[]):
    for i in range(vocab_size):
        ele_tmp = [random.random()-0.5 for i in range(embedded_dim)]
        ele_inter = math.sqrt(sum([ele**2 for ele in ele_tmp]))
        value.append([0.5*ele/ele_inter for ele in ele_tmp])
    with open(filename) as fd:
        while True:
            line = fd.readline()
            line = line.strip()
            if not line:
                break
            elems = line.split(' ')
            id = int(elems[0])
            embedding = [float(ele) for ele in elems[1:]]
            inter = math.sqrt(sum([ele**2 for ele in embedding]))
            embedding = [ele/(inter) for ele in embedding]
            value[id] = embedding
    return np.array(value)


def sparse_tensor_to_dense(sparse_tensor, width, sparse_values, default = 0):
    ###
    dense_tensor_shape = sparse_tensor.dense_shape
    return tf.cond(tf.equal(tf.shape(sparse_tensor.dense_shape)[0], 3),
                   lambda: tf.sparse_to_dense(sparse_indices = sparse_tensor.indices,
                                              output_shape = [dense_tensor_shape[0], dense_tensor_shape[1], width],
                                              sparse_values = sparse_values,
                                              default_value = default),
                   lambda: tf.sparse_to_dense(sparse_indices = sparse_tensor.indices,
                                              output_shape = [dense_tensor_shape[0], width],
                                              sparse_values = sparse_values,
                                              default_value = default)
                    )


def format_features(features):  ## 使用q_t_neib_sparse标记[query, title, neighbor1, neighbor2, ..., neighbor8]
    q_t_neib_sparse = [features['q'], features['sna'], features['qiq1_t'], features['qiq1_q'], features['qiq2_t'], features['qiq2_q'], features['iqi1_q'], features['iqi1_t'], features['iqi2_q'], features['iqi2_t']]
    ## word -> id (sparse -> val -> ids -> sparse)
    feature_name_unpad = ['q_unpad', 'sna_unpad', 'qiq1_t_unpad', 'qiq1_q_unpad', 'qiq2_t_unpad', 'qiq2_q_unpad', 'iqi1_q_unpad', 'iqi1_t_unpad', 'iqi2_q_unpad', 'iqi2_t_unpad']
    feature_name_dense = ['q', 'sna', 'qiq1_t', 'qiq1_q', 'qiq2_t', 'qiq2_q', 'iqi1_q', 'iqi1_t', 'iqi2_q', 'iqi2_t']
    for i in range(len(q_t_neib_sparse)):
        text_ids = tf.reshape(q_t_neib_sparse[i].values, [-1])
        text_word2ids = char_input_fn.word2ids(text_ids)
        text_ids = tf.reshape(text_word2ids, [-1])
        text_sparse = tf.SparseTensor(indices=q_t_neib_sparse[i].indices, values=tf.cast(text_ids, tf.int64), dense_shape=q_t_neib_sparse[i].dense_shape)
        features[feature_name_unpad[i]] = text_sparse
        if i in {0, 3, 5, 6, 8}:
            text_ids_pad = tf.sparse_slice(sp_input=text_sparse, start=[0,0,0], size=[text_sparse.dense_shape[0], text_sparse.dense_shape[1], model_config.query_size])
            text_ids_dense = sparse_tensor_to_dense(sparse_tensor=text_ids_pad, width=model_config.query_size, sparse_values=text_ids_pad.values)
            features[feature_name_dense[i]] = text_ids_dense
        else:
            text_ids_pad = tf.sparse_slice(sp_input=text_sparse, start=[0,0,0], size=[text_sparse.dense_shape[0], text_sparse.dense_shape[1], model_config.title_size])
            text_ids_dense = sparse_tensor_to_dense(sparse_tensor=text_ids_pad, width=model_config.title_size, sparse_values=text_ids_pad.values)
            features[feature_name_dense[i]] = text_ids_dense
    return features


def network_fn(features, labels, mode, params):
    """Build Model function f(x) for Estimator."""
    # ------build feature-------
    feature_name_dense = ['q', 'sna', 'qiq1_t', 'qiq1_q', 'qiq2_t', 'qiq2_q', 'iqi1_q', 'iqi1_t', 'iqi2_q', 'iqi2_t']
    feature_name_unpad = ['q_unpad', 'sna_unpad', 'qiq1_t_unpad', 'qiq1_q_unpad', 'qiq2_t_unpad', 'qiq2_q_unpad', 'iqi1_q_unpad', 'iqi1_t_unpad', 'iqi2_q_unpad', 'iqi2_t_unpad']    
    q_t_neib_ids = []
    q_t_neib_ids_sparse = []
    for i in range(len(feature_name_dense)):
        if i in {0, 3, 5, 6, 8}:    ## query_size
            q_t_neib_ids.append(tf.reshape(features[feature_name_dense[i]], [-1, model_config.query_size]))
        else:                       ## title_size
            q_t_neib_ids.append(tf.reshape(features[feature_name_dense[i]], [-1, model_config.title_size]))
    for i in range(len(feature_name_unpad)):
        q_t_neib_ids_sparse.append(features[feature_name_unpad[i]])

    # ------grams embedding-------
    def grams_embedding(word_ids, word_ids_sparse, word_len, var_scope="text"):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("embedding"):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    word2vector = word2vec_initiliar(model_config.vocab_file, model_config.vocab_size, model_config.embedding_dim)
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], initializer=tf.constant_initializer(word2vector),  trainable=True)
                else:
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], trainable=True)
                embedded_chars = tf.nn.embedding_lookup(embedded_weight, word_ids)
                embedded_chars_mean = tf.nn.embedding_lookup_sparse(embedded_weight, word_ids_sparse, None, combiner='mean')
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                pooled_outputs = []
                pooled_outputs.append(tf.reshape(embedded_chars, [-1, word_len, model_config.embedding_dim]))
        return pooled_outputs, embedded_chars_mean
    # ------sequence embedding-------
    def sequence_embedding(word_ids_sparse, var_scope="text"):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("embedding"):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    word2vector = word2vec_initiliar(model_config.vocab_file, model_config.vocab_size, model_config.embedding_dim)
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], initializer=tf.constant_initializer(word2vector),  trainable=True)
                else:
                    embedded_weight = tf.get_variable("weight", [model_config.vocab_size, model_config.embedding_dim], trainable=True)
                embedded_chars_mean = tf.nn.embedding_lookup_sparse(embedded_weight, word_ids_sparse, None, combiner='mean')
        return embedded_chars_mean

    query_embedded, query_embedding = grams_embedding(q_t_neib_ids[0], q_t_neib_ids_sparse[0], model_config.query_size)
    title_embedded, title_embedding = grams_embedding(q_t_neib_ids[1], q_t_neib_ids_sparse[1], model_config.title_size)
    # ------inner product-------
    query_title_inner = []
    position_embedded = tf.get_variable("position_weight", [1, model_config.title_size, model_config.embedding_dim], trainable=True)
    query_position_embedded, _ = tf.split(position_embedded, [model_config.query_size, model_config.title_size-model_config.query_size], 1)
    title_position_embedded = position_embedded
    for query_ele in query_embedded:
        for title_ele in title_embedded:
            query_ele = query_ele + query_position_embedded
            title_ele = title_ele + title_position_embedded
            inner_product = tf.expand_dims(calc_inner_product(query_ele, title_ele), -1)
            query_title_inner.append(inner_product)
    query_title_inner = tf.concat(query_title_inner, axis=-1)
    h_pool_flat = tf.reshape(query_title_inner, [-1, model_config.query_size*model_config.title_size*len(query_embedded)*len(title_embedded)])
    
    # ------gnn-------
    # ------aggregation and fusion (mean, attention)-------
    neigb_embedding = []
    for i in range(len(q_t_neib_ids_sparse)-2):
        neigb_embedding.append(sequence_embedding(q_t_neib_ids_sparse[i+2]))
    # ------query's metapath embedding (qiq)-------
    query_mp1_embedding = tf.stack([query_embedding, neigb_embedding[0], neigb_embedding[1]], axis=1)
    query_mp1_embedding = tf.reduce_mean(query_mp1_embedding, axis=1)
    query_mp2_embedding = tf.stack([query_embedding, neigb_embedding[2], neigb_embedding[3]], axis=1)
    query_mp2_embedding = tf.reduce_mean(query_mp2_embedding, axis=1)
    dims = query_mp1_embedding.shape[1]
    query_mp_att_vec = tf.get_variable('query_mp_att_vec', [2*dims, 1], trainable=True)
    query_mp1_import = tf.nn.leaky_relu(tf.matmul(tf.concat([query_embedding, query_mp1_embedding], axis=1), query_mp_att_vec))
    query_mp2_import = tf.nn.leaky_relu(tf.matmul(tf.concat([query_embedding, query_mp2_embedding], axis=1), query_mp_att_vec))
    query_mp_import_norm = tf.nn.softmax(tf.concat([query_mp1_import, query_mp2_import], axis=1), axis=1)
    query_mp1_import_norm = tf.tile(tf.expand_dims(query_mp_import_norm[:,0], 1), [1,dims])
    query_mp2_import_norm = tf.tile(tf.expand_dims(query_mp_import_norm[:,1], 1), [1,dims])
    query_mp_embedding = tf.stack([tf.multiply(query_mp1_import_norm, query_mp1_embedding), tf.multiply(query_mp2_import_norm, query_mp2_embedding)], axis=1)
    query_mp_embedding = tf.nn.relu(tf.reduce_sum(query_mp_embedding, axis=1))
    query_mp_w = tf.get_variable('query_mp_w', [dims, dims], trainable=True)
    query_mp_embedding = tf.nn.relu(tf.matmul(query_mp_embedding, query_mp_w))
    # ------title's metapath embedding (iqi)-------
    title_mp1_embedding = tf.stack([title_embedding, neigb_embedding[4], neigb_embedding[5]], axis=1)
    title_mp1_embedding = tf.reduce_mean(title_mp1_embedding, axis=1)
    title_mp2_embedding = tf.stack([title_embedding, neigb_embedding[6], neigb_embedding[7]], axis=1)
    title_mp2_embedding = tf.reduce_mean(title_mp2_embedding, axis=1)
    title_mp_att_vec = tf.get_variable('title_mp_att_vec', [2*dims, 1], trainable=True)
    title_mp1_import = tf.nn.leaky_relu(tf.matmul(tf.concat([title_embedding, title_mp1_embedding], axis=1), title_mp_att_vec))
    title_mp2_import = tf.nn.leaky_relu(tf.matmul(tf.concat([title_embedding, title_mp2_embedding], axis=1), title_mp_att_vec))
    title_mp_import_norm = tf.nn.softmax(tf.concat([title_mp1_import, title_mp2_import], axis=1), axis=1)
    title_mp1_import_norm = tf.tile(tf.expand_dims(title_mp_import_norm[:,0], 1), [1,dims])
    title_mp2_import_norm = tf.tile(tf.expand_dims(title_mp_import_norm[:,1], 1), [1,dims])
    title_mp_embedding = tf.stack([tf.multiply(title_mp1_import_norm, title_mp1_embedding), tf.multiply(title_mp2_import_norm, title_mp2_embedding)], axis=1)
    title_mp_embedding = tf.nn.relu(tf.reduce_sum(title_mp_embedding, axis=1))
    title_mp_w = tf.get_variable('title_mp_w', [dims, dims], trainable=True)
    title_mp_embedding = tf.nn.relu(tf.matmul(title_mp_embedding, title_mp_w))
    # ------concat all 5 embeddings-------
    h_pool_flat = tf.concat([h_pool_flat, query_embedding, title_embedding, query_mp_embedding, title_mp_embedding], axis=1)
    # ------nn full connect layers------
    with tf.variable_scope("nn-layers"):
        layers = model_config.deep_layers
        predict_scores = h_pool_flat
        for i, layer_size in enumerate(layers + [1]):
            predict_scores = tf.layers.dense(inputs=predict_scores,
                                             units=layer_size,
                                             activation=tf.nn.relu if i < len(layers) else None,
                                             use_bias=True,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                             trainable=True
                                             )
    predict_scores = tf.reshape(predict_scores, [-1])
    end_points["logits"] = predict_scores
    end_points["probabilities"] = tf.nn.sigmoid(predict_scores)
    return predict_scores, end_points


def loss_fn(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


def calc_focal_loss(labels, logits, gamma=2):
    prob = tf.nn.sigmoid(logits)
    pos_weight = tf.identity(tf.pow(1-prob, gamma))
    neg_weight = tf.identity(tf.pow(prob, gamma))
    sum_weight = tf.reduce_sum(tf.multiply(pos_weight, labels) + tf.multiply(neg_weight, 1-labels))
    pos_weight = tf.clip_by_value(tf.divide(pos_weight, sum_weight+1e-10), 0.1, 4)
    neg_weight = tf.clip_by_value(tf.divide(neg_weight, sum_weight+1e-10), 0.1, 4)
    pos = -tf.multiply( tf.stop_gradient(pos_weight), tf.multiply(labels, tf.log( prob ) ))
    neg = -tf.multiply( tf.stop_gradient(neg_weight), tf.multiply(1-labels, tf.log( 1-prob )) )
    return tf.reduce_sum(pos+neg)


def calc_inner_product(matrix_1, matrix_2):
    '''
    :param matrix_1: [None, length_1, size]
    :param matrix_2: [None, length_2, size]
    :return: [None, length_1, length_2]
    '''
    return tf.matmul(matrix_1, matrix_2, transpose_b=True)


def get_optimizer(learning_rate):
    #return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #return tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    #return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #return tf.train.FtrlOptimizer(learning_rate)
    #return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    return tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #return optimizer.MaskedAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def model_fn(features, labels, mode, params):
    """Build Model function f(x) for Estimator."""
    learning_rate = model_config.learning_rate
    gpus_list = model_config.gpus_list
    if len(gpus_list)==0 and len(params.get('gpus_list',"")) > 1 and mode == tf.estimator.ModeKeys.TRAIN:
        gpus_list = [int(ele) for ele in params['gpus_list'].split(",")]

    features = format_features(features)

    spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits, end_point = network_fn(features, labels, mode, model_config)
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=end_point)
    else:
        # ------bulid optimizer------
        optimizer = get_optimizer(learning_rate)
        global_step = tf.train.get_or_create_global_step()

        # tower model
        eval_logits = []
        total_loss = 0
        if len(gpus_list) > 1 and False:
            features_split = [{} for i in range(len(gpus_list))]
            labels_split = []
            for name in features.keys():
                value = features[name]
                if isinstance(value, tf.SparseTensor):
                    value_split = tf.sparse_split(sp_input=value, num_split=len(gpus_list), axis=0)
                else:
                    value_split = tf.split(value, num_or_size_splits=len(gpus_list), axis=0)

                for i in range(len(gpus_list)):
                    features_ele = features_split[i]
                    features_ele[name] = value_split[i]
                    features_split[i] = features_ele
                if len(labels_split) <= 0:
                    labels_split = tf.split(labels, num_or_size_splits=len(gpus_list), axis=0)

            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(len(gpus_list)):
                    gpu_index = gpus_list[i]
                    with tf.device('/gpu:%d' % gpu_index):
                        with tf.name_scope('%s_%d' % ("tower-gpu", gpu_index)) as scope:
                            # model and loss
                            logits, end_points = network_fn(features_split[i], labels_split[i], mode, model_config)
                            losses = loss_fn(labels_split[i], logits)
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                            updates_op = tf.group(*update_ops)
                            with tf.control_dependencies([updates_op]):
                                total_loss += losses / len(gpus_list)
                            # reuse var
                            tf.get_variable_scope().reuse_variables()
                            # grad compute
                            grads = optimizer.compute_gradients(losses)
                            tower_grads.append(grads)
                            # for eval metric ops
                            eval_logits.append(end_points)

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
            train_op = tf.group(apply_gradient_op, grads)

        else:
            logits, end_points = network_fn(features, labels, mode, params)
            total_loss = loss_fn(labels, logits)
            eval_logits.append(end_points)
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        y_predict = tf.concat([ele["probabilities"] for ele in eval_logits], 0)
        y_predict_2 = tf.where(y_predict < 0.5, x=tf.zeros_like(y_predict), y=tf.ones_like(y_predict))
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_predict_2),
                "auc": tf.metrics.auc(labels, y_predict)
            }
        if mode == tf.estimator.ModeKeys.TRAIN:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metrics)

    return spec


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
