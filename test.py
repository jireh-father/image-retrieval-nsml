import numpy as np
from scipy.spatial import distance

query_vecs = np.random.random_sample([195, 32])
import tensorflow as tf


def l2_normalize(v, axis):
    norm = np.linalg.norm(v, axis=axis)
    print(norm)
    if norm == 0:
        return v
    return v / norm


r = np.linalg.norm([[5, 10], [4, 3]])
print(r)
# reference_vecs = l2_normalize([[5, 10], [4, 3]], axis=None)
# print(reference_vecs)
# reference_vecs = l2_normalize([[5, 10], [4, 3]], axis=1)
# print(reference_vecs)
# # reference_vecs = l2_normalize([[5, 10], [4, 3]], axis=0)
# # print(reference_vecs)
#
# import sys
# sys.exit()
# from sklearn import preprocessing
#
# data_nor = preprocessing.normalize([[5, 100], [4, 3]], norm='l2', axis=0)
#
# print(data_nor)
# data_nor = preprocessing.normalize([[5, 100], [4, 3]], norm='l2', axis=1)
# print(data_nor)
ret = tf.Variable([[5., 10., 2.], [4., 3., 8.]])
embeddings = tf.nn.l2_normalize(ret, axis=1)
embeddings2 = tf.nn.l2_normalize(ret, axis=0)
embeddings3 = tf.nn.l2_normalize(ret)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(embeddings))
print(sess.run(embeddings2))
print(sess.run(embeddings3))
# embeddings3 = tf.nn.l2_normalize(ret, axis=2)
# print(embeddings, embeddings2)
import sys

sys.exit()

reference_vecs = np.random.random_sample([1127, 32])

ret1 = distance.cdist(query_vecs, reference_vecs, 'euclidean')

query_vecs = np.random.random_sample([195, 32])

reference_vecs = np.random.random_sample([1127, 32])

ret2 = distance.cdist(query_vecs, reference_vecs, 'euclidean')

print(ret.shape)
print(ret)
sys.exit()
dot_product = np.matmul(query_vecs, np.transpose(reference_vecs))
print(dot_product.shape)
square_norm = np.diagonal(dot_product)
print(square_norm.shape)
print(np.expand_dims(square_norm, 1).shape)
print(np.expand_dims(square_norm, 0).shape)
distances = np.expand_dims(square_norm, 1) - 2.0 * dot_product + np.expand_dims(square_norm, 0)
print(distances)
sim_matrix = np.maximum(distances, 0.0)
print(sim_matrix.shape)
