import logging
import os
import sys
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from cdcgan import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import torch.optim as optim
from PIL import Image
from StyleGAN.model import StyleBased_Generator, Discriminator

torch.manual_seed(0)

def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def imsave(tensor, i):
    grid = tensor[0]
    grid.clamp_(-1, 1).add_(1).div_(2)
    # Add 0.5 after normalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(f'samples/sample-iter{i}.png')
    

class GModel():
    def __init__(self, model):
        self.model = model
    def __call__(self, x):
        return self.model(x)[0]


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()

        # init StyleGAN
        self.generator = StyleBased_Generator(8, 256, 4).to('cuda')
        self.discriminator = Discriminator().to('cuda')
        self.g_optim = optim.Adam([{
            'params': self.generator.convs.parameters(),
            'lr'    : 0.001
        }, {
            'params': self.generator.to_rgbs.parameters(),
            'lr'    : 0.001
        }], lr=0.001, betas=(0.0, 0.99))
        self.g_optim.add_param_group({
            'params': self.generator.fcs.parameters(),
            'lr'    : 0.001 * 0.01,
            'mul'   : 0.01
        })
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))

        # checkpoint = torch.load('checkpoint/trained.pth')

        # self.generator.load_state_dict(checkpoint['generator'])
        # self.discriminator.load_state_dict(checkpoint['discriminator'])
        # self.g_optim.load_state_dict(checkpoint['g_optim'])
        # self.d_optim.load_state_dict(checkpoint['d_optim'])


        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train_gan(self,sample, step, iteration=0, startpoint=0, used_sample=0,
         d_losses = [], g_losses = [], alpha=0):
            
            generator=self.generator
            discriminator=self.discriminator
            g_optim=self.g_optim
            d_optim=self.d_optim
            real_image = sample[0]
            
            # D Module ---
            # Train discriminator first
            set_grad_flag(discriminator, True)
            set_grad_flag(generator, False)
            discriminator.zero_grad()
            
            # Real image predict & backward
            # We only implement non-saturating loss with R1 regularization loss
            real_image.requires_grad = True
            real_predict = self.discriminator(real_image, step, alpha)
            real_predict = F.softplus(-real_predict).mean()
            real_predict.backward(retain_graph=True)

            grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
            grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty_real = 10 / 2 * grad_penalty_real
            grad_penalty_real.backward()
            
            # Generate latent code
            latent_w1 = [torch.randn((32, 256), device='cuda')]
            latent_w2 = [torch.randn((32, 256), device='cuda')]

            noise_1 = []
            noise_2 = []
            for m in range(step + 1):
                size = 4 * 2 ** m # Due to the upsampling, size of noise will grow
                noise_1.append(torch.randn((32, 1, size, size), device='cuda'))
                noise_2.append(torch.randn((32, 1, size, size), device='cuda'))
            
            
            fake_image = generator(latent_w1, step, alpha, noise_1)
            fake_predict = discriminator(fake_image, step, alpha)

            fake_predict = nn.functional.softplus(fake_predict).mean()
            fake_predict.backward()
            
            if iteration % 10 == 0:
                d_losses.append((real_predict + fake_predict).item())
            
            # D optimizer step
            d_optim.step()
            
            # Avoid possible memory leak
            del grad_penalty_real, grad_real, fake_predict, real_predict, fake_image, real_image, latent_w1
                    
            # G module ---
            # Due to DGR, train generator
            generator.zero_grad()
            set_grad_flag(discriminator, False)
            set_grad_flag(generator, True)
            
            fake_image = generator(latent_w2, step, alpha, noise_2)
            fake_predict = discriminator(fake_image, step, alpha)
            fake_predict = nn.functional.softplus(-fake_predict).mean()
            fake_predict.backward(retain_graph=True)
            g_optim.step()

            if iteration % 10 == 0:
                g_losses.append(fake_predict.item())
                imsave(fake_image.data.cpu(), iteration)
            
            fi = fake_image.detach().clone()
            # Avoid possible memory leak
            del fake_predict, fake_image, latent_w2
            
            if iteration % 1000 == 0:
                # Save the model every 1000 iterations
                torch.save({
                    'generator'    : generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optim'      : g_optim.state_dict(),
                    'd_optim'      : d_optim.state_dict(),
                    'parameters'   : (step, iteration, startpoint, used_sample, alpha),
                    'd_losses'     : d_losses,
                    'g_losses'     : g_losses
                }, 'checkpoint/trained_gan.pth')
                print(f'Model successfully saved.')
                
            return d_losses, g_losses, fi, fake_predict

        

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")


        d_losses=[];g_losses=[]
        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                d_losses, g_losses, fi, fp = self.train_gan((images, torch.tensor([0]*len(images))), step=1)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss+fp).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
