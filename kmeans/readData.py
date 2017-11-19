import numpy as np
import pickle
from scipy import misc
from os import listdir
from os.path import isfile, join

def readImageData(shape=(200,200), amt=-1):
  PATH = 'data/photos/'
  onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
  images = []
  metadata = []
  i = 0 
  if amt != -1:
    onlyfiles = onlyfiles[:amt]

  for f in onlyfiles:
    if i % 100 == 0:
      print i
    i+=1
    a = misc.imread(PATH+f, mode = "L")
    aspect_ratio = float(a.shape[0]) / a.shape[1]
    if 0.5 < aspect_ratio < 2.0:
      metadata.append({
        'filename': f,
        'dimensions': a.shape,
      })
      a = misc.imresize(a, shape)
      a = a.flatten()
      images.append(a)
  return np.vstack(images), metadata

def readSerializedData():
  PHOTOS_PATH = 'data/transformedPhotos.npy'
  METADATA_PATH = 'data/photoMetadata.p'
  X = np.load(PHOTOS_PATH)
  metadata = pickle.load(open(METADATA_PATH, 'rb'))

  return X, metadata
