import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random

class Discriminator(nn.Module):
  def __init__(self, ngpu, nc, ndf, backup_file):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # input is (nc) x 64 x 64
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )
    self.apply(weights_init)
    if backup_file != '':
      self.load_state_dict(torch.load(backup_file))

  def forward(self, input):
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)

    return output.view(-1, 1).squeeze(1)

class Generator(nn.Module):
  def __init__(self, ngpu, nc, nz, ngf, backup_file):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(     nz, ngf * 64, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 64),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.PixelShuffle(2),
      nn.BatchNorm2d(ngf * 16),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.PixelShuffle(2),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.PixelShuffle(2),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )
    self.apply(weights_init)
    if backup_file != '':
      self.load_state_dict(torch.load(backup_file))

  def forward(self, input):
    if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
      output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
    else:
      output = self.main(input)
    return output

# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def train(opt, netD, netG, dataloader, nz):
  criterion = nn.BCELoss()

  input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
  noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
  fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
  label = torch.FloatTensor(opt.batchSize)
  real_label = 1
  fake_label = 0

  if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

  fixed_noise = Variable(fixed_noise)

  # setup optimizer
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

  print 'Starting training'
  for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
      flipper = random.random() < .1

      ############################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      # train with real
      netD.zero_grad()
      real_cpu, _ = data
      batch_size = real_cpu.size(0)
      if opt.cuda:
        real_cpu = real_cpu.cuda()
      input.resize_as_(real_cpu).copy_(real_cpu)
      if not flipper:
        label.resize_(batch_size).fill_(real_label)
      else:
        label.resize_(batch_size).fill_(fake_label)
      inputv = Variable(input)
      labelv = Variable(label)

      output = netD(inputv)
      errD_real = criterion(output, labelv)
      errD_real.backward()
      D_x = output.data.mean()

      # train with fake
      noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
      noisev = Variable(noise)
      fake = netG(noisev)
      if not flipper:
        labelv = Variable(label.fill_(fake_label))
      else:
        labelv = Variable(label.fill_(real_label))
      output = netD(fake.detach())
      errD_fake = criterion(output, labelv)
      errD_fake.backward()
      D_G_z1 = output.data.mean()
      errD = errD_real + errD_fake
      optimizerD.step()

      ############################
      # (2) Update G network: maximize log(D(G(z)))
      ###########################
      netG.zero_grad()
      labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
      output = netD(fake)
      errG = criterion(output, labelv)
      errG.backward()
      D_G_z2 = output.data.mean()
      optimizerG.step()

      if i % 10 == 0:
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, opt.niter, i, len(dataloader),
             errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
      if i % 100 == 0:
        vutils.save_image(real_cpu,
            '%s/real_samples.png' % opt.outf,
            normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.data,
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
