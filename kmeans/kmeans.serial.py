import numpy as numpy
import sklearn.cluster as cl
import readData as rd
import outputClusters
import sys

if len(sys.argv) < 3:
  raise Exception('required arguments: numImages, numClusters')

numImages = int(sys.argv[1]) # -1 means all images
numClusters = int(sys.argv[2])

X, metadata = rd.readSerializedData()
if numImages != -1:
  X = X[:numImages]
  metadata = metadata[:numImages]
clusters = cl.KMeans(n_clusters=numClusters, random_state = 0, verbose=1)
X_clusters = clusters.fit_predict(X)

cluster_map = {}
for i in xrange(len(X_clusters)):
  if X_clusters[i] not in cluster_map:
    cluster_map[X_clusters[i]] = []
  cluster_map[X_clusters[i]].append(i)

medians = []
clusterCenters = clusters.cluster_centers_
for i in xrange(numClusters):
  a = clusterCenters[i].reshape((200,200)).astype('uint8')
  medians.append(a)

params = {
  'numImages': numImages,
  'numClusters': numClusters,
}
outputClusters.generatePage(cluster_map, metadata, medians, 'kmeans', extraData=params)
