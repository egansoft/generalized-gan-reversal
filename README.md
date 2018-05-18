# Generalized Latent Variable Recovery for Generative Adversarial Networks
The Generator of a Generative Adversarial Network (GAN) is trained to transform latent vectors drawn from a prior distribution into realistic looking photos.  These latent vectors have been shown to encode information about the content of their corresponding images.  Projecting input images onto the latent space of a GAN is non-trivial,  but  previous  work  has  successfully  performed  this  task  for  latent  spaces  with  a  uniform prior.  We extend these techniques to latent spaces  with a  Gaussian  prior,  and  demonstrate our technique's effectiveness.  

Our GAN was trained on food photos from the [Yelp dataset](https://www.yelp.com/dataset) and uses the DCGAN architecture in PyTorch.

## Generator Samples

<p align="center">
  <a href="samples/gan-long.png">
    <img src="samples/gan-long.png" width="450" /><br />
    Standard DCGAN
  </a>
</p>

<p align="center">
  <a href="samples/gan-ps.png">
    <img src="samples/gan-ps.png" width="450" /><br />
    Pixel Shuffle DCGAN
  </a>
</p>

<p align="center">
  <a href="samples/gan-soft.png">
    <img src="samples/gan-soft.png" width="450" /><br />
    Soft Label DCGAN
  </a>
</p>

<p align="center">
  <a href="samples/latent-walk.png">
    <img src="samples/latent-walk.png" width="450" /><br />
    Latent Vector Interpolation (SLERP)
  </a>
</p>

## Authors
- [Nicholas Egan](http://nicholasegan.me)
- [Jeffrey Zhang](https://github.com/jzhang12)
- [Kevin Shen](https://github.com/kevinshen11)

## Attributions
Code
- [DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)
- [Recovery for Uniform Prior](https://github.com/yxlao/pytorch-reverse-gan)

Data
- [Yelp Dataset](https://www.yelp.com/dataset)
