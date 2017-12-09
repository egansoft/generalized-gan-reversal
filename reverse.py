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
import time

def zNorm(z):
  vec = z.data.view(100)
  return torch.dot(vec, vec)

def gaussPdf(z):
  vec = z.data.view(100)
  return torch.sum(-(vec ** 2))/2 - 100.*math.log(2*math.pi)/2

def reverse_z(netG, x, z, z_approx, opt, clip, params):
  assert clip in ['disabled', 'tn', 'hard', 'logistic']
  mse_loss = nn.MSELoss()
  mse_loss_ = nn.MSELoss()

  if opt.cuda:
    mse_loss = mse_loss.cuda()
    mse_loss_ = mse_loss_.cuda()
    z_approx = z_approx.cuda()

  z_approx = Variable(z_approx)
  z_approx.requires_grad = True
  optimizer_approx = optim.Adam([z_approx], lr=opt.lr, betas=(opt.beta1, 0.999))

  torch.manual_seed(opt.manualSeed)
  if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

  lastXLoss = float('inf')
  clips = 0
  startTime = time.time()
  for i in xrange(opt.niter):
    x_approx = netG(z_approx)
    mse_x = mse_loss(x_approx, x)

    '''
    xLoss = mse_x.data[0]
    if abs(xLoss - lastXLoss) < 1e-10:
      break
    lastXLoss = xLoss
    '''

    optimizer_approx.zero_grad()
    mse_x.backward()
    optimizer_approx.step()

    # clipping
    zData = z_approx.data.abs()
    thresh = torch.rand(100).cuda()
    thresh.resize_as_(zData)
    if clip == 'hard':
      # p(clip) = [[|x| > cutoff]]
      cutoff = params
      amt = z_approx.data[zData > cutoff].size()
      z_approx.data[zData > cutoff] = random.normalvariate(0, 1)

    if clip == 'tn':
      # p(clip all) = Nt(cutoff)/Nt(x)
      cutoff = params
      zData.clamp_(0, cutoff)
      zData.pow_(2) # x^2
      zData.mul_(-0.5) # -x^2/2
      zData.exp_() # exp(-x^2/2)
      zData.reciprocal_() # 1/exp(-x^2/2)
      zData.mul_(-np.exp(-np.power(cutoff, 2)/2)) # -nt(cutoff)/exp(-x^2/2)
      zData.add_(1) # 1 - np(cutoff)/exp(-x^2/2)
      zData.pow_(1./opt.expectedIter) # (1 - np(cutoff)/exp(-x^2/2))^(1/E)
      zData.mul_(-1) # -(1 - np(cutoff)/exp(-x^2/2))^(1/E)
      zData.add_(1) # 1 - (1 - np(cutoff)/exp(-x^2/2))^(1/E)

      amt = z_approx.data[zData > thresh].size()
      z_approx.data[zData > thresh] = random.normalvariate(0, 1)
    if clip == 'logistic':
      # p(clip all) = 1/(1 + exp(-a(|x| - b)))
      b, a = params
      zData.add_(-b) # x-b
      zData.mul_(-a) # -a(x-b)
      zData.exp_() # exp(-a(x-b))
      zData.add_(1) # 1 + exp(-a(x-b))
      zData.reciprocal_() # 1/(1 + exp(-a(x-b)))
      zData.mul_(-1) # -1/(1 + exp(-a(x-b)))
      zData.add_(1) # 1 - 1/(1 + exp(-a(x-b)))
      zData.pow_(1./opt.expectedIter) # (1 - 1/(1 + exp(-a(x-b))))^(1/E)
      zData.mul_(-1) # -(1 - 1/(1 + exp(-a(x-b))))^(1/E)
      zData.add_(1) # 1 - (1 - 1/(1 + exp(-a(x-b))))^(1/E)
    
      amt = z_approx.data[zData > thresh].size()
      z_approx.data[zData > thresh] = random.normalvariate(0, 1)
    if clip == 'disabled':
      amt = ()

    if len(amt) > 0:
      clips += amt[0]

  xLoss = mse_loss(x_approx, x).data[0]
  zLoss = mse_loss_(z_approx, z).data[0]
  probZ = gaussPdf(z_approx)
  duration = int(time.time() - startTime)
  print("{}: mse_x: {}, MSE_z: {}, P(z): {}, T: {}, clips: {}, params: {}, time: {}".format(clip, xLoss, zLoss, probZ, i, clips, params, duration))
  #vutils.save_image(x_approx.data, 'output/reverse/x_approx.png', normalize=True)

  recoveries = np.array([1 if zLoss < thresh else 0 for thresh in [1e-4, 1e-3, 1e-2, 1e-1, 1e0]])
  return z_approx, zLoss, recoveries

def reverse_gan(opt):
  # load netG and fix its weights
  netG = gan.Generator(opt.ngpu, opt.nc, opt.nz, opt.ngf, opt.netG)
  for param in netG.parameters():
    param.requires_grad = False
  netG = netG.cuda()

  grid = [
    ('hard', 2.5), ('hard', 3), ('hard', 3.5),
    ('tn', 2.5), ('tn', 2.75), ('tn', 3), ('tn', 3.25), ('tn', 3.5),
    ('logistic', (2,2)), ('logistic', (2,3)), ('logistic', (2,4)),
    ('logistic', (2.5,2)), ('logistic', (2.5,3)), ('logistic', (2.5,4)),
    ('logistic', (3,2)), ('logistic', (3,3)), ('logistic', (3,4))
  ]
  numTrials = len(grid)
  wins = np.zeros(numTrials)
  sigWins = np.zeros(numTrials)
  sigTotal = np.zeros(numTrials)
  recoveries = np.zeros((numTrials + 1, 5))
  totalErrors = np.zeros(numTrials + 1)

  for i in xrange(opt.niter):
    z = Variable(torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1))
    z.data.resize_(1, opt.nz, 1, 1)
    z = z.cuda()
    probZ = gaussPdf(z)
    x = netG(z)
    print i, 'P(z) =', probZ
    opt.manualSeed = random.randint(1, 10000)
    z_fixed = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)

    zApproxDisabled, zErrorDisabled, recoveryDisabled = reverse_z(netG, x, z, z_fixed.clone(), opt, 'disabled', ())
    recoveries[0] += recoveryDisabled
    totalErrors[0] += zErrorDisabled
    for j, (clippingMethod, params) in enumerate(grid):
      zApprox, zError, recovery = reverse_z(netG, x, z, z_fixed.clone(), opt, clippingMethod, params)
      recoveries[j+1] += recovery
      totalErrors[j+1] += zError

      signif = abs(zError / zErrorDisabled) > 2 or abs(zError / zErrorDisabled) < 0.5
      if signif:
        sigTotal[j] += 1
      if zError < zErrorDisabled:
        wins[j] += 1
        if signif:
          sigWins[j] += 1
    print wins
    print sigWins, sigTotal
    print totalErrors/(i+1)
    print recoveries
    print

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clip', default='disabled',
            help='disabled|standard|stochastic|gaussian')
  parser.add_argument('--niter', type=int, default=20000,
            help='number of epochs to run for')
  parser.add_argument('--lr', type=float, default=0.01,
            help='learning rate, default=0.01')
  parser.add_argument('--beta1', type=float, default=0.5,
            help='beta1 for adam. default=0.5')
  parser.add_argument('--netG', default='sampleGparams.pth',
            help="path to netG (to continue training)")

  opt = parser.parse_args()
  opt.cuda = True
  opt.expectedIter = 20000
  opt.nz, opt.nc, opt.ngf, opt.ngpu = 100, 3, 64, 1

  print(opt)

  # process arguments
  if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run "
        "with --cuda")

  if opt.cuda:
    cudnn.benchmark = True  # turn on the cudnn autotuner

  reverse_gan(opt)
