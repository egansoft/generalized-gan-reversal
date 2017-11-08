import numpy as np
from scipy import misc
from os import listdir
from os.path import isfile, join

def readImageData(shape = (200,200), amt=-1):
  PATH = 'data/photos/'
  onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
  images = []
  i = 0 
  if amt != -1:
    onlyfiles = onlyfiles[:amt]

  for f in onlyfiles:
    if i % 100 == 0:
      print i
    i+=1
    a = misc.imread(mypath+f, mode = "L")
    aspect_ratio = float(a.shape[0]) / a.shape[1]
    if 0.5 < aspect_ratio < 2.0:
      a = misc.imresize(a, shape).flatten()
      images.append(a)
  return np.vstack(images)

def readSerializedData():
  PATH = 'data/transformedPhotos.npy'
  return np.load(PATH)
