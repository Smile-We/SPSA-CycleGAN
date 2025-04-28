import os

import numpy as np
import itertools
import time
import datetime

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image
import matplotlib.image as mpimg

# from Kmean import organize_cyclegan_direc
# from SPCN.Stain_Sep import multi_sep
from utils import ImageDataset, save_checkpoint, LambdaLR, \
    initialize_conv_weights_normal, ReplayBuffer
from sa_cyclegan import *

cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=8,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=16,
    img_size=256,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=9,
    lambda_cyc=10,
    lambda_id=5.0,
)

def train(
        Gen_BA,
        Gen_AB,
        Disc_A,
        Disc_B,
        train_dataloader,
        n_epochs,
        criterion_cycle,
        lambda_cyc,
        criterion_GAN,
        criterion_cycle_GAN,
        optimizer_G,
        fake_A_buffer,
        fake_B_buffer,
        clear_output,
        optimizer_Disc_A,
        optimizer_Disc_B,
        Tensor,
        sample_interval,
        lambda_id,
        save_path
):
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))
            A_outline = Variable(batch["A_outline"].type(Tensor))
            B_outline = Variable(batch["B_outline"].type(Tensor))
            # Adversarial ground truths i.e. target vectors
            # 1 for real images and 0 for fake generated images
            valid = Variable(
                Tensor(np.ones((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                Tensor(np.zeros((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            #########################
            #  Generators
            #########################

            Gen_AB.train()
            Gen_BA.train()

            optimizer_G.optimizer.zero_grad()

            # # Identity loss
            # # First pass real_A images to the Genearator, that will generate A-domains images
            # loss_id_A = criterion_identity(Gen_BA(real_A), real_A)
            #
            # # Then pass real_B images to the Genearator, that will generate B-domains images
            # loss_id_B = criterion_identity(Gen_AB(real_B), real_B)
            #
            # loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN losses for GAN_AB
            fake_B = Gen_AB(real_A)

            loss_GAN_AB = criterion_GAN(Disc_B(fake_B), valid)
            # gray scale loss
            loss_cycle_GAN_A = criterion_cycle_GAN(A_outline, transforms.Grayscale()(fake_B))

            # GAN losses for GAN_BA
            fake_A = Gen_BA(real_B)

            loss_GAN_BA = criterion_GAN(Disc_A(fake_A), valid)
            # gray scale loss
            loss_cycle_GAN_B = criterion_cycle_GAN(B_outline, transforms.Grayscale()(fake_A))

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2 + (loss_cycle_GAN_A + loss_cycle_GAN_B) * 5

            # Cycle Consistency losses
            reconstructed_A = Gen_BA(fake_B)

            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)

            reconstructed_B = Gen_AB(fake_A)

            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2


            loss_G = loss_GAN + loss_cycle
            # + lambda_id * loss_identity

            loss_G.backward()
            
            optimizer_G.step(epoch)
            optimizer_G.optimizer.step()

            #########################
            #  Train Discriminator A
            #########################

            optimizer_Disc_A.optimizer.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)

            if epoch > n_epochs / 2:
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            else:
                fake_A_ = fake_A

            loss_fake = criterion_GAN(Disc_A(fake_A_.detach()), fake)

            loss_Disc_A = (loss_real + loss_fake) / 2

            loss_Disc_A.backward()

            optimizer_Disc_A.step(epoch)
            optimizer_Disc_A.optimizer.step()

            #########################
            #  Train Discriminator B
            #########################

            optimizer_Disc_B.optimizer.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_B(real_B), valid)

            # Fake loss (on batch of previously generated samples)
            if epoch > n_epochs / 2:
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            else:
                fake_B_ = fake_B

            loss_fake = criterion_GAN(Disc_B(fake_B_.detach()), fake)

            loss_Disc_B = (loss_real + loss_fake) / 2

            loss_Disc_B.backward()

            optimizer_Disc_B.step(epoch)
            optimizer_Disc_B.optimizer.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2

            batches_done = epoch * len(train_dataloader) + i
            batches_left = n_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    # loss_identity.item(),
                    time_left,
                )
            )

        if epoch>=10 and (epoch%10==0 or epoch==n_epochs-1):
            save_checkpoint(Gen_BA, optimizer_G, filename=save_path+"gen_" + str(epoch) + "_b.pth.tar")
            save_checkpoint(Gen_AB, optimizer_G, filename=save_path+"gen_" + str(epoch) + "_a.pth.tar")
            save_checkpoint(Disc_A, optimizer_Disc_A,
                            filename=save_path+"disc_" + str(epoch) + "_a.pth.tar")
            save_checkpoint(Disc_B, optimizer_Disc_B,
                            filename=save_path+"disc_" + str(epoch) + "_b.pth.tar")


def cn_cyclegan_train(root_path, model_mode, save_path):

    transforms_ = [
        transforms.Resize(int(hp.img_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop((hp.img_size, hp.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_dataloader = DataLoader(
        ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_=transforms_),
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=1,
    )

    criterion_GAN = torch.nn.MSELoss()

    criterion_cycle = torch.nn.L1Loss()

    '''This is the gray scale loss to fix the sri problem'''
    criterion_cycle_GAN = torch.nn.L1Loss()

    criterion_identity = torch.nn.L1Loss()

    input_shape = (hp.channels, hp.img_size, hp.img_size)

    Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks, mode=model_mode)
    Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks, mode=model_mode)

    Disc_A = Discriminator(input_shape)
    Disc_B = Discriminator(input_shape)

    if cuda:
        Gen_AB = Gen_AB.cuda()
        Gen_BA = Gen_BA.cuda()
        Disc_A = Disc_A.cuda()
        Disc_B = Disc_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    Gen_AB.apply(initialize_conv_weights_normal)
    Gen_BA.apply(initialize_conv_weights_normal)

    Disc_A.apply(initialize_conv_weights_normal)
    Disc_B.apply(initialize_conv_weights_normal)

    fake_A_buffer = ReplayBuffer()

    fake_B_buffer = ReplayBuffer()

    optimizer_G = torch.optim.Adam(
        itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
        lr=hp.lr,
        betas=(hp.b1, hp.b2),
    )
    optimizer_Disc_A = torch.optim.Adam(Disc_A.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))

    optimizer_Disc_B = torch.optim.Adam(Disc_B.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step
    )

    lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_A,
        lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
    )

    lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_B,
        lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
    )

    train(
        Gen_BA=Gen_BA,
        Gen_AB=Gen_AB,
        Disc_A=Disc_A,
        Disc_B=Disc_B,
        train_dataloader=train_dataloader,
        n_epochs=hp.n_epochs,
        criterion_cycle=criterion_cycle,
        lambda_cyc=hp.lambda_cyc,
        criterion_GAN=criterion_GAN,
        criterion_cycle_GAN=criterion_cycle_GAN,
        optimizer_G=lr_scheduler_G,
        fake_A_buffer=fake_A_buffer,
        fake_B_buffer=fake_B_buffer,
        clear_output=clear_output,
        optimizer_Disc_A=lr_scheduler_Disc_A,
        optimizer_Disc_B=lr_scheduler_Disc_B,
        Tensor=Tensor,
        sample_interval=hp.sample_interval,
        lambda_id=hp.lambda_id,
        save_path=save_path,
    )


if __name__ == "__main__":
    root = "/home/SPSA_CycleGAN/"
    model_mode = ['att_plus', 'no_att']
    save_path = root+f'param/new_Camelyon/grey_att/' 
    data_path = root+"data/Camelyon/train/"
    cn_cyclegan_train(data_path, model_mode=model_mode[0], save_path=save_path)
