# SS-GEAR

## AML Major Project
Contributors: Aaditya Baranwal; Mohit Mathuria; Nakul Sharma

OS Version: Linux: Ububtu 20.04

esrgan-> This folder contains the code for ESRGAN, the file inf_foler.py scales the images. The file gen_images.py uses the StableDiffusion and GIT to augment the images, and save them offline.

SimCLR-> Training and accompanying scripts for running our method which exploits stable diffusion for augmentations and train SimCLR network to learn hidden representations.

SimCLR GAN-> Training and accompanying scripts for running our method which exploits styleGAN for augmentations and train SimCLR network.

The approach is explained in `report.pdf`.

to run training scripts, run the following command after goining to respective method's directory:
```
python run.py --gpu-index <CUDA device index>
```
## References:
Mentioned in `report.pdf`
