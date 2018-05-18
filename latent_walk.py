import gan
import numpy as np
import sys
import torch
import torchvision.utils as vutils
from torch.autograd import Variable


def slerp(p0, p1, t):
  omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def sample_spherical(npoints, ndim=3):
  vec = np.random.randn(ndim, npoints)
  vec /= np.linalg.norm(vec, axis=0)
  return vec

def interpolate(U):
  u = point / np.linalg.norm(U)
  v = np.resize(sample_spherical(1, 100), 100)
  V = v * np.linalg.norm(U)
  w = -(u + v) / np.linalg.norm(u + v)
  W = w * np.linalg.norm(U)
  interpolation = []
  print(U)
  print(V)
  print(W)
  slerps = np.linspace(0, 0.9999, 21)
  slerps2 = np.linspace(0, 0.9999, 22)
  for i in slerps:
    interpolation.append(slerp(U, V, i))
  for i in slerps:
    interpolation.append(slerp(V, W, i))
  for i in slerps2:
    interpolation.append(slerp(W, U, i))
  interpolation = np.concatenate(interpolation, axis = 0)
  return interpolation.reshape(64,100,1,1)


if len(sys.argv) > 1:
  savedParams = sys.argv[1]
else:
  savedParams = 'sampleGparams.pth'
if len(sys.argv) > 2:
  outputImagePath = sys.argv[2]
else:
  outputImagePath = 'output/walk.png'
if len(sys.argv) > 3:
  numSamples = sys.argv[3]
else:
  numSamples = 64

ngpu, nc, nz, ngf = 1, 3, 100, 64, # lol
G = gan.Generator(ngpu, nc, nz, ngf, savedParams)

noise = torch.FloatTensor(numSamples, nz, 1, 1)
noise.normal_(0, 1)

point = sample_spherical(1, 100)
point = 0.2 * np.resize(point, 100)
SLERP = interpolate(point)


noise = torch.from_numpy(SLERP)

noisev = Variable(noise).float()
fake = G(noisev)
vutils.save_image(fake.data, outputImagePath, normalize=True)

