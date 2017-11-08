import numpy as np
import readData

X = readData.readImageData()
np.save('data/transformedPhotos.npy', X)
