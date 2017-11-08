import numpy as np
import readData

X = readData.readImageData("data/photos/")
np.save('data/transformedPhotos.npy', X)
