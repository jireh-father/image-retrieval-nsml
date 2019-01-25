"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model import nets_factory

slim = tf.contrib.slim


def _configure_learning_rate(num_samples_per_epoch, global_step, cf):
    """Configures the learning rate.
  
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
  
    Returns:
      A `Tensor` representing the learning rate.
  
    Raises:
      ValueError: if
    """
    # Note: when num_clones is > 1, this will actually have each clone to go
    # over each epoch cf.num_epochs_per_decay times. This is different
    # behavior from sync replicas and is expected to produce different results.
    decay_steps = int(num_samples_per_epoch * cf.num_epochs_per_decay /
                      cf.batch_size)

    if cf.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(cf.learning_rate,
                                          global_step,
                                          decay_steps,
                                          cf.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif cf.learning_rate_decay_type == 'fixed':
        return tf.constant(cf.learning_rate, name='fixed_learning_rate')
    elif cf.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(cf.learning_rate,
                                         global_step,
                                         decay_steps,
                                         cf.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         cf.learning_rate_decay_type)


def _configure_optimizer(learning_rate, cf):
    """Configures the optimizer used for training.
  
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
  
    Returns:
      An instance of an optimizer.
  
    Raises:
      ValueError: if cf.optimizer is not recognized.
    """
    if cf.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=cf.adadelta_rho,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=cf.adagrad_initial_accumulator_value)
    elif cf.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=cf.adam_beta1,
            beta2=cf.adam_beta2,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=cf.ftrl_learning_rate_power,
            initial_accumulator_value=cf.ftrl_initial_accumulator_value,
            l1_regularization_strength=cf.ftrl_l1,
            l2_regularization_strength=cf.ftrl_l2)
    elif cf.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=cf.momentum,
            name='Momentum',
            use_nesterov=cf.use_nesterov)
    elif cf.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=cf.rmsprop_decay,
            momentum=cf.rmsprop_momentum,
            epsilon=cf.opt_epsilon)
    elif cf.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized' % cf.optimizer)
    return optimizer


def build_slim_model(is_training, images, params, class_cnt=None):
    wd = 0.
    if hasattr(params, "weight_decay"):
        wd = params.weight_decay
    if params.use_crossentropy:
        num_classes = [int(params.embedding_size), class_cnt]
    else:
        num_classes = int(params.embedding_size)
    model_f = nets_factory.get_network_fn(params.model_name, num_classes, wd,
                                          is_training=is_training)
    out, end_points = model_f(images)

    return out, end_points


def _get_variables_to_train(cf):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if cf.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in cf.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def train_op_fun(total_loss, global_step, num_examples, cf):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if cf.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            cf.moving_average_decay, global_step)
        update_ops.append(variable_averages.apply(moving_average_variables))

    lr = _configure_learning_rate(num_examples, global_step, cf)
    tf.summary.scalar('learning_rate', lr)
    opt = _configure_optimizer(lr, cf)
    variables_to_train = _get_variables_to_train(cf)
    grads = opt.compute_gradients(total_loss, variables_to_train)
    grad_updates = opt.apply_gradients(grads, global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_op = tf.identity(total_loss, name='train_op')

    return train_op


def build_model(features, labels=None, cf=None, is_training=True, num_examples=None, global_step=None, class_cnt=None):
    images = features

    embeddings, end_points = build_slim_model(is_training, images, cf, class_cnt)

    if not is_training:
        return embeddings

    if cf.l2norm:
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if cf.triplet_strategy == "batch_all":
        loss = batch_all_triplet_loss(labels, embeddings, margin=cf.margin, squared=cf.squared)
    elif cf.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=cf.margin, squared=cf.squared)
    elif cf.triplet_strategy == "semihard":
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings, margin=cf.margin)
    elif cf.triplet_strategy == "cluster":
        loss = tf.contrib.losses.metric_learning.cluster_loss(labels, embeddings, 1.0)
    elif cf.triplet_strategy == "contrastive":
        pass
    elif cf.triplet_strategy == "lifted_struct":
        loss = tf.contrib.losses.metric_learning.lifted_struct_loss(labels, embeddings, margin=cf.margin)
    elif cf.triplet_strategy == "npairs":
        tf.contrib.losses.metric_learning.npairs
        pass
    elif cf.triplet_strategy == "npairs_multilabel":
        pass
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(cf.triplet_strategy))

    vars = tf.trainable_variables()

    if cf.use_crossentropy:
        loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, class_cnt),
                                                                          logits=end_points['Logits2']))

    loss += tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * cf.weight_decay

    train_op = train_op_fun(loss, global_step, num_examples, cf)

    return loss, end_points, train_op, embeddings
