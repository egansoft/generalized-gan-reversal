import numpy as numpy
import sklearn.cluster as cl
import readData as rd

X = rd.readSerializedData()
clusters = cl.KMeans(n_clusters = 4, random_state = 0)
X_clusters = clusters.fit_predict(X)

cluster_map = {}
for i in xrange(len(X_clusters)):
  if X_clusters[i] not in cluster_map:
    cluster_map[X_clusters[i]] = []
  cluster_map[X_clusters[i]].append(i)
