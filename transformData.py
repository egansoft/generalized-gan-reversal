import numpy as np
import pickle
import readData

X, metadata = readData.readImageData()
np.save('data/transformedPhotos.npy', X)
pickle.dump(metadata, open('data/photoMetadata.p', 'wb'))
