import numpy as numpy
import sklearn.cluster as cl
import transformData as rd

X = rd.readData("data/photos/")
clusters = cl.KMeans(n_clusters = 4, random_state = 0)
X_clusters = clusters.fit_predict(X)

cluster_map = {}
for i in xrange(len(X_clusters)):
  print i
  if X_clusters[i] not in cluster_map:
    cluster_map[X_clusters[i]] = []
  cluster_map[X_clusters[i]].append(i)
