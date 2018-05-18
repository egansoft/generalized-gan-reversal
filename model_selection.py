import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import gan

def evalModel(dCheckpoint, gCheckpoint, dataloader):
  D = gan.Discriminator(ngpu, nc, ndf, dCheckpoint).cuda()
  G = gan.Generator(ngpu, nc, nz, ngf, gCheckpoint).cuda()

  batch_size = dataloader.batch_size
  input = torch.FloatTensor(batch_size, 3, 64, 64).cuda()
  noise = torch.FloatTensor(batch_size, 100, 1, 1).cuda()

  dMean = 0.
  gMean = 0.
  nIterations = 10
  for i, data in enumerate(dataloader, 0):
    if i >= 10:
      break

    noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
    noisev = Variable(noise)
    fake = G(noisev)
    output = D(fake.detach())
    gCor = output.data.mean()
    gMean += gCor

  #dMean /= nIterations
  gMean /= nIterations
  #print dMean, gMean
  #print 'score:', (dMean - gMean), '   (', dMean, gMean, ')'
  return dMean, gMean

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="data/", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)

opt = parser.parse_args()
print opt

imageSize = 64
batchSize = 1000
ngpu = 1
nz = 100
ngf = 64
ndf = 64
nc = 3
dataset = dset.ImageFolder(root=opt.dataroot,
  transform=transforms.Compose([
    transforms.Scale(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
  shuffle=True, num_workers=int(opt.workers))

models = [
  ('short', 13),
  ('soft', 25),
  ('long', 50),
]

for modelName, modelAmt in models:
  results = []
  for i in xrange(modelAmt):
    dPath = 'output/gan-samples-short/netD_epoch_11.pth'
    gPath = 'output/gan-samples-' + modelName + '/netG_epoch_' + str(i) + '.pth'
    dMean, gMean = evalModel(dPath, gPath, dataloader)
    score = gMean
    print modelName + str(i), score
    results.append((score, i))
  results.sort(reverse=True)
  print results
