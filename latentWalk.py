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

def reverse_slerp(p0, p1, t):
  omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def sample_spherical(npoints, ndim=3):
  vec = np.random.randn(ndim, npoints)
  vec /= np.linalg.norm(vec, axis=0)
  return vec

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


#Partial Interpolation
'''
#Generate array of slerp points
slerps = np.linspace(0, 0.99, 64)
interpolation = []
other_vec = sample_spherical(1, 100)
other_vec = 10 * np.resize(other_vec,100)
print(other_vec)
for i in slerps:
  interpolation.append(slerp(other_vec, np.ones(100), i))
interpolation = np.concatenate(interpolation, axis = 0)
SLERP = interpolation.reshape(64,100,1,1)
'''

#Full Interpolation
slerps = np.linspace(0, 0.99, 21)
slerps2 = np.linspace(0, 0.99, 22)
interpolation = []

def ETFFrame(n):
  vec1, vec2, vec3 = np.array([0,1]), np.array([-np.sqrt(3)/2, -1/2]), np.array([np.sqrt(3)/2, -1/2])
  vec1 = 10 * np.append(np.append(np.zeros(n%98), vec1), np.zeros(98 - n%98))
  vec2 = 10 * np.append(np.append(np.zeros(n%98), vec2), np.zeros(98 - n%98))
  vec3 = 10 * np.append(np.append(np.zeros(n%98), vec3), np.zeros(98 - n%98))
  return(vec1, vec2, vec3)

vec1, vec2, vec3 = ETFFrame(17)

for i in slerps:
  interpolation.append(slerp(vec1, vec2, i))
for i in slerps:
  interpolation.append(slerp(vec2, vec3, i))
for i in slerps2:
  interpolation.append(slerp(vec3, vec1, i))
interpolation = np.concatenate(interpolation, axis = 0)
SLERP = interpolation.reshape(64,100,1,1)


noise = torch.from_numpy(SLERP)

noisev = Variable(noise).float()
fake = G(noisev)
vutils.save_image(fake.data, outputImagePath, normalize=True)

