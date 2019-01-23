import tensorflow as tf
import numpy as np

images_ph = tf.random_uniform([10, 2, 2, 1], 0., 20.)
# images_ph = tf.placeholder(tf.float32, [10, 2, 2, 1],
#                            name="inputs")
labels_ph = tf.placeholder(tf.int32, [10], name="labels")
print(labels_ph)
indices_equal = tf.cast(tf.eye(tf.shape(labels_ph)[0]), tf.bool)
indices_not_equal = tf.logical_not(indices_equal)
labels_equal = tf.equal(tf.expand_dims(labels_ph, 0), tf.expand_dims(labels_ph, 1))
mask = tf.cast(tf.logical_and(indices_not_equal, labels_equal), tf.int32)
print(mask)

print(tf.reduce_sum(mask, axis=0))
# ret = tf.squeeze(tf.where(tf.reduce_sum(mask, axis=0) > 0))
ret = tf.where(tf.reduce_sum(mask, axis=0) > 0)
imgs = tf.concat([tf.gather_nd(images_ph, ret), images_ph[:2]], axis=0)
print("cc", imgs)
lbls = tf.concat([tf.gather_nd(labels_ph, ret), labels_ph[:2]], axis=0)
print("cc", lbls)


def train_pre_process(filename):
    return filename


imgs = imgs[:2]
dataset = tf.data.Dataset.from_tensor_slices(imgs)
dataset = dataset.map(train_pre_process, num_parallel_calls=4)
# dataset.
if cf.use_pair_sampling:
    dataset = dataset.batch(cf.sampling_buffer_size)
    dataset = dataset.prefetch(cf.sampling_buffer_size * 8)
else:
    dataset = dataset.batch(cf.batch_size)
    dataset = dataset.prefetch(cf.batch_size * 8)

iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

print(ret)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run([ret, tf.reduce_sum(mask, axis=0), tf.gather_nd(labels_ph, ret), tf.gather_nd(images_ph, ret)],
                  feed_dict={labels_ph: np.array([1, 2, 33, 1, 2, 34, 99, 444, 3, 4])})

print(result)
print(result[3].shape)
