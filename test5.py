import numpy as np
from sklearn.neighbors import BallTree

np.random.seed(0)
queries = []
db = []
for i in range(10):
    queries.append("a/b/%d.jpg" %i)
    db.append("a/c/%d.jpg" % i)
db = np.array(db)


X = np.random.random((10, 3))  # 10 points in 3 dimensions
Q = np.random.random((10, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)
dist, index_list = tree.query(Q, k=3)
print(index_list)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors
retrieval_results = {}

for (i, ind) in enumerate(index_list):
    query = queries[i].split('/')[-1].split('.')[0]
    print(ind)
    print(type(ind))
    print(db[ind])
    ranked_list = [k.split('/')[-1].split('.')[0] for k in db[ind]]  # ranked list

    retrieval_results[query] = ranked_list
print('done')

print(list(zip(range(len(retrieval_results)), retrieval_results.items())))
