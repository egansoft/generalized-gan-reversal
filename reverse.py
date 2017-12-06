import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import gan
import numpy as np
from scipy.stats import norm
import math

def zNorm(z):
  vec = z.data.view(100)
  return torch.dot(vec, vec)

def gaussPdf(z):
  vec = z.data.view(100)
  return torch.sum(-(vec ** 2))/2 - 100.*math.log(2*math.pi)/2

def reverse_z(netG, x, z, opt, lam=0, clip='disabled'):
  """
  Estimate z_approx given G and G(z).

  Args:
    netG: nn.Module, generator network.
    x: Variable, G(z).
    opt: argparse.Namespace, network and training options.
    z: Variable, the ground truth z, ref only here, not used in recovery.
    clip: Although clip could come from of `opt.clip`, here we keep it
        to be more explicit.
  Returns:
    Variable, z_approx, the estimated z value.
  """
  # sanity check
  assert clip in ['disabled', 'standard', 'stochastic', 'probabilistic']

  # loss metrics
  mse_loss = nn.MSELoss()
  mse_loss_ = nn.MSELoss()

  # init tensor
  if opt.z_distribution == 'uniform':
    z_approx = torch.FloatTensor(1, opt.nz, 1, 1).uniform_(-1, 1)
  elif opt.z_distribution == 'normal':
    z_approx = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
  else:
    raise ValueError()

  # transfer to gpu
  if opt.cuda:
    mse_loss.cuda()
    mse_loss_.cuda()
    z_approx = z_approx.cuda()

  # convert to variable
  z_approx = Variable(z_approx)
  z_approx.requires_grad = True

  # optimizer
  optimizer_approx = optim.Adam([z_approx], lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=lam)

  # train
  lastXLoss = float('inf')
  for i in xrange(opt.niter):

    x_approx = netG(z_approx)
    mse_x = mse_loss(x_approx, x)
    if i % 1000 == 0:
      mse_z = mse_loss_(z_approx, z)
      zL2 = zNorm(z_approx)
      probZ = gaussPdf(z_approx)
      xLoss, zLoss = mse_x.data[0], mse_z.data[0]

      if abs(xLoss - lastXLoss) < 1e-4:
        break
      lastXLoss = xLoss
      #print("[Iter {}] mse_x: {}, MSE_z: {}, W: {}, P(z): {}".format(i, xLoss, zLoss, zL2, probZ))

    # bprop
    optimizer_approx.zero_grad()
    mse_x.backward()
    optimizer_approx.step()

    # clipping
    if clip == 'standard':
      z_approx.data[z_approx.data > 1] = 1
      z_approx.data[z_approx.data < -1] = -1
    if clip == 'stochastic':
      z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
      z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)
    if clip == 'probabilistic':
      for i in range(100):
        prob = norm.pdf(z_approx.data[0, i, 0, 0])
        if random.random() < math.exp(-1000 * prob):
          z_approx.data[0, i, 0, 0] = np.random.normal(0, 1)

  if i == opt.niter-1:
    print 'maxed',

  xLoss = mse_loss(x_approx, x).data[0]
  zLoss = mse_loss_(z_approx, z).data[0]
  probZ = gaussPdf(z_approx)
  print("{}: mse_x: {}, MSE_z: {}, P(z): {}".format(lam, xLoss, zLoss, probZ))
  vutils.save_image(x_approx.data, 'output/reverse/x_approx.png', normalize=True)

  #print list(z_approx.data.view(100))
  return z_approx, np.array([xLoss, zLoss, probZ])

def reverse_gan(opt):
  # load netG and fix its weights
  netG = gan.Generator(opt.ngpu, opt.nc, opt.nz, opt.ngf, opt.netG)
  for param in netG.parameters():
    param.requires_grad = False

  lams = [0]
  epochs = 8
  allLosses = np.zeros((len(lams), epochs, 3))
  for i in xrange(epochs):
    z = Variable(torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1))
    z.data.resize_(1, opt.nz, 1, 1)
    netG.cuda()
    z = z.cuda()
    probZ = gaussPdf(z)
    x = netG(z)
    #vutils.save_image(x.data, 'output/reverse/x.png', normalize=True)
    #print(z.cpu().data.numpy().squeeze())
    print 'z', i, probZ

    for li, lam in enumerate(lams):
      z_approx, losses = reverse_z(netG, x, z, opt, lam=lam, clip=opt.clip)
      allLosses[li, i] = losses

  print
  for li, lam in enumerate(lams):
    avgLosses = np.sum(allLosses[li] / epochs, axis=0)
    xLoss, zLoss, probZ = tuple(avgLosses)
    var = np.sum((allLosses[li] - avgLosses) ** 2, axis=0) / epochs
    xVar, zVar, probVar = tuple(var)
    print("{}: mse_x: {}, MSE_z: {}, P(z): {}".format(lam, xLoss, zLoss, probZ))
    print("   var mse_x: {}, MSE_z: {}, P(z): {}".format(xVar, zVar, probVar))

  #print(z_approx.cpu().data.numpy().squeeze())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clip', default='disabled',
            help='disabled|standard|stochastic|gaussian')
  parser.add_argument('--z_distribution', default='normal',
            help='uniform | normal')
  parser.add_argument('--nz', type=int, default=100,
            help='size of the latent z vector')
  parser.add_argument('--nc', type=int, default=3,
            help='number of channels in the generated image')
  parser.add_argument('--ngf', type=int, default=64)
  parser.add_argument('--niter', type=int, default=10000,
            help='number of epochs to train for')
  parser.add_argument('--lam', type=float, default=0.0)
  parser.add_argument('--lr', type=float, default=0.01,
            help='learning rate, default=0.0002')
  parser.add_argument('--beta1', type=float, default=0.5,
            help='beta1 for adam. default=0.5')
  parser.add_argument('--ngpu', type=int, default=1,
            help='number of GPUs to use')
  parser.add_argument('--netG', default='sampleGparams.pth',
            help="path to netG (to continue training)")
  parser.add_argument('--manualSeed', type=int, help='manual seed')
  parser.add_argument('--profile', action='store_true',
            help='enable cProfile')

  opt = parser.parse_args()
  opt.cuda = True
  print(opt)

  # process arguments
  if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run "
        "with --cuda")

  if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
  random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)
  if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True  # turn on the cudnn autotuner
    # torch.cuda.set_device(1)

  reverse_gan(opt)

