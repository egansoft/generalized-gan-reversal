import numpy as np
from scipy import misc
from os import listdir
from os.path import isfile, join

def readData(mypath, shape = (200,200)):
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  X = []
  i = 0
  for f in onlyfiles[:500]:
    print i
    i+=1
    a = misc.imread(mypath+f, mode = "L")
    aspect_ratio = float(a.shape[0]) / a.shape[1]
    if 0.5 < aspect_ratio < 2.0:
      a = misc.imresize(a, shape)
      if len(X) == 0:
        X = a.flatten()
      else:
        X = np.vstack((X, a.flatten()))
  return X
