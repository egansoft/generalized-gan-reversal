import gan
import sys
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

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
noisev = Variable(noise)
fake = G(noisev)
vutils.save_image(fake.data, outputImagePath, normalize=True)
