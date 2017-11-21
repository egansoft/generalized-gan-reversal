import gan
import sys
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

savedParams = sys.argv[1]
outputImagePath = sys.argv[2]

ngpu, nc, nz, ngf, batch_size = 1, 3, 100, 64, 64 # lol
G = gan.Generator(ngpu, nc, nz, ngf, savedParams)

noise = torch.FloatTensor(batch_size, nz, 1, 1)
noise.normal_(0, 1)
noisev = Variable(noise)
fake = G(noisev)
vutils.save_image(fake.data, outputImagePath, normalize=True)
