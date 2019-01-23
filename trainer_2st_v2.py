import os
import tensorflow as tf
from model import model_fn
import glob
import util
from datetime import datetime
import time
import numpy as np
from preprocessing import preprocessing_factory
import nsml
import tfrecorder_builder as tb
from nsml import DATASET_PATH
import pickle

slim = tf.contrib.slim

fl = tf.app.flags
fl.DEFINE_string('save_dir', 'experiments/test', '')
fl.DEFINE_integer('num_preprocessing_threads', 8, '')
fl.DEFINE_integer('log_every_n_steps', 2, 'The frequency with which logs are print.')
fl.DEFINE_integer('save_interval_epochs', 1, '')

fl.DEFINE_string('mode', 'train', 'submit일때 해당값이 test로 설정됩니다.')
fl.DEFINE_string('iteration', '0',
                 'fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
fl.DEFINE_integer('pause', 0, 'model 을 load 할때 1로 설정됩니다.')
fl.DEFINE_boolean('test', False, '')

#######################
# Dataset Flags #
#######################

fl.DEFINE_string('model_name', 'inception_resnet_v2', '')
fl.DEFINE_string('preprocessing_name', 'inception', '')
fl.DEFINE_integer('batch_size', 64, '')
fl.DEFINE_integer('eval_batch_size', 128, '')
fl.DEFINE_boolean('use_pair_sampling', True, '')
fl.DEFINE_integer('sampling_buffer_size', 600, '')
fl.DEFINE_integer('shuffle_buffer_size', 700, '')  # default 500
fl.DEFINE_integer('train_image_channel', 3, '')
fl.DEFINE_integer('train_image_size', 299, '')  # pnasnet_large 331, inception_resnet 299, resnet 224
fl.DEFINE_integer('max_number_of_epochs', 200, '')
fl.DEFINE_integer('keep_checkpoint_max', 200, '')

#######################
# Triplet #
#######################
fl.DEFINE_integer('embedding_size', 128, '')
fl.DEFINE_string('triplet_strategy', 'semihard', '')
fl.DEFINE_float('margin', 0.5, '')
fl.DEFINE_boolean('squared', False, '')
fl.DEFINE_boolean('l2norm', True, '')

######################
# Optimization Flags #
######################

# fl.DEFINE_float('weight_decay', 0.00004, '')
# fl.DEFINE_float('weight_decay', 0.0002, '')
fl.DEFINE_float('weight_decay', 0.00004, '')
fl.DEFINE_string('optimizer', 'momentum', '"adadelta", "adagrad", "adam",''"ftrl", "momentum", "sgd"  "rmsprop".')
fl.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
fl.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')
fl.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
fl.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
fl.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
fl.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
fl.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
fl.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
fl.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
fl.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
# fl.DEFINE_float('momentum', 0.5, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
fl.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
fl.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################
fl.DEFINE_string('learning_rate_decay_type', 'exponential', '"fixed", "exponential",'' or "polynomial"')
# fl.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# fl.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
# fl.DEFINE_float('learning_rate', 0.000754, 'Initial learning rate.')
fl.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
# fl.DEFINE_float('learning_rate', 0.000754, 'Initial learning rate.')
# fl.DEFINE_float('end_learning_rate', 0.0001, 'The minimal end learning rate used by a polynomial decay.')
fl.DEFINE_float('end_learning_rate', 0.00001, 'The minimal end learning rate used by a polynomial decay.')
fl.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
fl.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
fl.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.')

#####################
# Fine-Tuning Flags #
#####################
# fl.DEFINE_string('nsml_checkpoint', '73', '')
# fl.DEFINE_string('nsml_session', 'jireh_family/ir_ph1_v2/251', '')
fl.DEFINE_string('nsml_checkpoint', None, '')
fl.DEFINE_string('nsml_session', None, '')
fl.DEFINE_boolean('fine_tuning', True, '')
fl.DEFINE_string('checkpoint_path', "./pretrained/inception_resnet_v2_2016_08_30.ckpt", '')
# fl.DEFINE_string('checkpoint_path', None, '')
# fl.DEFINE_string('checkpoint_exclude_scopes', None,
#                  'Comma-separated list of scopes of variables to exclude when restoring '
#                  'from a checkpoint.')
fl.DEFINE_string('checkpoint_exclude_scopes', "InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits",
                 'Comma-separated list of scopes of variables to exclude when restoring '
                 'from a checkpoint.')

fl.DEFINE_string('trainable_scopes', None, 'Comma-separated list of scopes to filter the set of variables to train.'
                                           'By default, None would train all the variables.')

pretrained_map = {
    "pnasnet_large": "https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz",
    "inception_resnet_v2": "http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz",
    "resnet_v2_152": "http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz",
    "nasnet_large": "https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz"
}

pretrained_filename_map = {
    "pnasnet_large": "model.ckpt",
    "nasnet_large": "model.ckpt",
    "inception_resnet_v2": "inception_resnet_v2_2016_08_30.ckpt",
    "resnet_v2_152": "resnet_v2_152.ckpt"
}

checkpoint_exclude_scopes_map = {
    "pnasnet_large": "aux_7/aux_logits/FC,final_layer/FC",
    "nasnet_large": "aux_7/aux_logits/FC,final_layer/FC",
    "inception_resnet_v2": "InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits",
    "resnet_v2_152": "resnet_v2_152/logits",
}


def bind_model(saver, sess, images_ph, embeddings_op, cf):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        saver.save(sess, dir_name + "/model.ckpt")
        print('model saved!')

    def load(file_path):
        saver.restore(sess, os.path.join(file_path, "model.ckpt"))
        print('model loaded!')

    def infer(queries, db):
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            eval_image_size = cf.train_image_size
            if cf.preprocessing_name is not None:
                image_preprocessing_fn = preprocessing_factory.get_preprocessing(cf.preprocessing_name,

                                                                                 is_training=False)
                image_decoded = image_preprocessing_fn(image_decoded, eval_image_size, eval_image_size)
            else:
                image = tf.cast(image_decoded, tf.float32)

                image = tf.image.central_crop(image, central_fraction=0.9)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_image_with_pad(image, cf.train_image_size, cf.train_image_size)
                image = tf.squeeze(image, [0])

                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)

                image = tf.divide(image, 255.0)
                image = tf.subtract(image, 0.5)
                image_decoded = tf.multiply(image, 2.0)

            return image_decoded

        dataset_queries = tf.data.Dataset.from_tensor_slices(queries)
        dataset_queries = dataset_queries.map(_parse_function)
        dataset_queries = dataset_queries.batch(cf.eval_batch_size)
        iterator = dataset_queries.make_one_shot_iterator()
        features = iterator.get_next()

        dataset_db = tf.data.Dataset.from_tensor_slices(db)
        dataset_db = dataset_db.map(_parse_function)
        dataset_db = dataset_db.batch(cf.eval_batch_size)
        iterator_db = dataset_db.make_one_shot_iterator()
        features_db = iterator_db.get_next()

        query_steps = int(len(queries) / cf.eval_batch_size)
        if len(queries) % cf.eval_batch_size > 0:
            query_steps += 1

        query_vecs = np.zeros((len(queries), int(cf.embedding_size)))

        for i in range(query_steps):
            query_imgs = sess.run(features)
            feed_dict = {images_ph: query_imgs}
            tmp_query_vecs = sess.run(embeddings_op, feed_dict=feed_dict)
            for j, tmp_qe in enumerate(tmp_query_vecs):
                query_vecs[i * cf.eval_batch_size + j] = tmp_qe

        db_steps = int(len(db) / cf.eval_batch_size)
        if len(db) % cf.eval_batch_size > 0:
            db_steps += 1

        reference_vecs = np.zeros((len(db), int(cf.embedding_size)))

        for i in range(db_steps):
            db_imgs = sess.run(features_db)
            feed_dict = {images_ph: db_imgs}
            tmp_db_vecs = sess.run(embeddings_op, feed_dict=feed_dict)
            for j, tmp_qe in enumerate(tmp_db_vecs):
                reference_vecs[i * cf.eval_batch_size + j] = tmp_qe

        print('inference start')

        # inference
        # caching db output, db inference

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(db, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    nsml.bind(save=save, load=load, infer=infer)


def \
  l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


if __name__ == '__main__':
    cf = fl.FLAGS

    tf.logging.set_verbosity(tf.logging.INFO)

    # tf.set_random_seed(123)
    if cf.mode == 'train':
        train_dataset_path = DATASET_PATH + '/train/train_data'
        # tb.make_tfrecords("ir_ph1_v2", "train", train_dataset_path, "./dataset/train/", 4, 3, False)
        tb.make_tfrecords("ir_ph1_v2", "train", train_dataset_path, "/tmp/igseo_tfrecord", 4, 3, False)

        files = glob.glob("./dataset/train/*_train*tfrecord")
        print(files)
        files.sort()
        assert len(files) > 0
        num_examples = util.count_records(files)
        global_step = tf.Variable(0, trainable=False)

        image_preprocessing_fn = None
        if cf.preprocessing_name:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(cf.preprocessing_name, is_training=True)


        def train_pre_process(example_proto):
            features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                        "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                        'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                        'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                        }

            parsed_features = tf.parse_single_example(example_proto, features)
            image = tf.image.decode_jpeg(parsed_features["image/encoded"], cf.train_image_channel)

            if image_preprocessing_fn is not None:
                image = image_preprocessing_fn(image, cf.train_image_size, cf.train_image_size)
            else:
                image = tf.cast(image, tf.float32)

                image = tf.image.central_crop(image, central_fraction=0.9)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_image_with_pad(image, cf.train_image_size, cf.train_image_size)
                image = tf.squeeze(image, [0])

                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)

                image = tf.divide(image, 255.0)
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)

            label = parsed_features["image/class/label"]
            return image, label


        steps_each_epoch = int(num_examples / cf.batch_size)
        if num_examples % cf.batch_size > 0:
            steps_each_epoch += 1
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(train_pre_process, num_parallel_calls=cf.num_preprocessing_threads)
        dataset = dataset.shuffle(cf.shuffle_buffer_size)
        dataset = dataset.repeat()
        if cf.use_pair_sampling:
            dataset = dataset.batch(cf.sampling_buffer_size)
            dataset = dataset.prefetch(cf.sampling_buffer_size * 8)
        else:
            dataset = dataset.batch(cf.batch_size)
            dataset = dataset.prefetch(cf.batch_size * 8)

        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
    else:
        num_examples = None
        global_step = None

    images_ph = tf.placeholder(tf.float32, [None, cf.train_image_size, cf.train_image_size, cf.train_image_channel],
                               name="inputs")
    labels_ph = tf.placeholder(tf.int32, [None], name="labels")
    if cf.mode == 'train':
        if cf.use_pair_sampling:
            loss_op, end_points, train_op, embeddings_op = model_fn.build_model(images_ph, labels_ph, cf, True,
                                                                                num_examples, global_step)
        else:
            loss_op, end_points, train_op, embeddings_op = model_fn.build_model(images, labels, cf, True, num_examples,
                                                                                global_step)
    else:
        embeddings_op = model_fn.build_model(images_ph, labels_ph, cf, is_training=False)

    if cf.mode == 'train':
        if cf.fine_tuning and cf.model_name in pretrained_map:
            import urllib.request
            import tarfile

            pretrained_url = pretrained_map[cf.model_name]
            os.makedirs("./pretrained")
            pretrained_filename = os.path.basename(pretrained_url)
            urllib.request.urlretrieve(pretrained_url, "./pretrained/" + pretrained_filename)

            tar = tarfile.open("./pretrained/" + pretrained_filename)
            tar.extractall("./pretrained")
            print("pretrained extracted")

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    epoch = 1
    steps = 1
    latest_epoch = 0
    if cf.mode == 'train' and (cf.nsml_checkpoint is None or cf.nsml_session is None):
        if cf.fine_tuning and cf.model_name in pretrained_map:
            checkpoint_path = "./pretrained/%s" % pretrained_filename_map[cf.model_name]
            exclusions = []
            checkpoint_exclude_scopes = checkpoint_exclude_scopes_map[cf.model_name]
            if checkpoint_exclude_scopes:
                exclusions = [scope.strip()
                              for scope in checkpoint_exclude_scopes.split(',')]
            variables_to_restore = []
            for var in slim.get_model_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        break
                else:
                    variables_to_restore.append(var)

            saver_for_restore = tf.train.Saver(var_list=variables_to_restore, max_to_keep=cf.keep_checkpoint_max)
            saver_for_restore.restore(sess, checkpoint_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=cf.keep_checkpoint_max)
    num_trained_images = 0

    bind_model(saver, sess, images_ph, embeddings_op, cf)

    if cf.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if cf.mode == 'train':
        bTrainmode = True
        if cf.nsml_checkpoint is not None and cf.nsml_session is not None:
            nsml.load(checkpoint=cf.nsml_checkpoint, session=cf.nsml_session)
        while True:
            try:
                start = time.time()

                if cf.use_pair_sampling:
                    print("pair sampling")
                    tmp_images, tmp_labels = sess.run([images, labels])
                    pair_indices = set()
                    single_index_map = {}
                    label_buffer = {}
                    for i, tmp_label in enumerate(tmp_labels):
                        if tmp_label in label_buffer:
                            pair_indices.add(i)
                            pair_indices.add(label_buffer[tmp_label])
                            if tmp_label in single_index_map:
                                del single_index_map[tmp_label]
                        else:
                            label_buffer[tmp_label] = i
                            single_index_map[tmp_label] = i
                    pair_indices = list(pair_indices)
                    print(len(pair_indices))
                    if len(pair_indices) > cf.batch_size:
                        pair_indices = pair_indices[:cf.batch_size]
                    elif len(pair_indices) < cf.batch_size:
                        pair_indices += list(single_index_map.values())[:cf.batch_size - len(pair_indices)]
                    batch_images = tmp_images[pair_indices]
                    batch_labels = tmp_labels[pair_indices]
                else:
                    print("not pair sampling")
                sampling_time = time.time() - start
                tmp_images = None
                tmp_labels = None
                start = time.time()
                if cf.use_pair_sampling:
                    feed_dict = {images_ph: batch_images, labels_ph: batch_labels}
                    loss, _ = sess.run([loss_op, train_op], feed_dict=feed_dict)
                else:
                    loss, _ = sess.run([loss_op, train_op])
                train_time = time.time() - start

                if steps % cf.log_every_n_steps == 0:
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print("[%s: %d epoch(%d/%d), %d steps] sampling time: %f, train time: %f, loss: %f" % (
                        now, epoch, steps % steps_each_epoch, steps_each_epoch, steps, sampling_time, train_time, loss))
                num_trained_images += cf.batch_size

                steps += 1
                if num_trained_images >= num_examples:
                    # nsml.report(summary=True, epoch=epoch, epoch_total=cf.max_number_of_epochs, loss=loss)

                    if cf.save_interval_epochs >= 1 and (
                      epoch - latest_epoch) % cf.save_interval_epochs == 0:
                        nsml.save(epoch)
                    if epoch >= cf.max_number_of_epochs:
                        break
                    epoch += 1
                    num_trained_images = 0
                if cf.test:
                    nsml.save(epoch)
                    break
            except tf.errors.OutOfRangeError:
                break
    print("end!!")
