AML Major Project
Contributors: Aaditya Baranwal; Mohit Mathuria; Nakul Sharma

OS Version: Linux: Ububtu 20.04
esrgan-> This folder contains the code for ESRGAN, the file inf_foler.py scales the images. The file gen_images.py uses the StableDiffusion and GIT to augment the images, and save them offline.
SimCLR-> Training and accompanying scripts for running our method which exploits stable diffusion for augmentations.
SimCLR GAN-> Training and accompanying scripts for running our method which exploits styleGAN for augmentations.

to run training scripts, run the following command:
python run.py --gpu-index <CUDA device index>
