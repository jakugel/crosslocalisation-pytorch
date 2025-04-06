# crosslocalisation_pytorch
PyTorch code for papers proposing cross-localisation, a simple yet powerful conditional StyleGAN2 architectural and training extension, to enable additional semi-supervised learning through data synthesis. 

In our research, significant improvements to OCT segmentation performance have been realised using GAN-based data augmentation techniques including the novel "cross-localisation" technique which additionally leverages information from unlabelled data samples to boost the diversity of labelled synthesised samples.

**Relevant papers**
1. Semi-supervised learning with cross-localisation in shared GAN latent space for enhanced OCT data augmentation (conference paper, part of Best of DICTA2022).
Link: https://www.nature.com/articles/s41598-019-49816-4

2. Enhanced OCT chorio-retinal segmentation in low-data settings with semi-supervised GAN augmentation using cross-localisation (Best of DICTA2022 extended journal paper).
Link: https://www.sciencedirect.com/science/article/pii/S1077314223002321

If the code and methods here are useful to you and aided in your research, please consider citing the papers above.


**Code**

This code is inspired by the following StyleGAN2 implementations:

https://blog.paperspace.com/implementation-stylegan2-from-scratch/

https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0
