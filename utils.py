import os

import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch
import glob
import random
from torch.utils.data import Dataset
from PIL import Image


def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        
        # self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        # self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        self.files = sorted(glob.glob(os.path.join(root, "*.*")))
        # print("self.files_B ", self.files_B)
        """ Will print below array with all file names
        ['/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-06-26 14:04:52.jpg',
        '/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-08-02 09:19:52.jpg',..]
        """

    def __getitem__(self, index):
        image_A = Image.open(self.files[index % len(self.files)])
        A_shape = np.array(image_A.size)
        A_name = os.path.basename(self.files[index % len(self.files)])
        
        B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]
        B_name = os.path.basename(B_path)
        if self.unaligned:
            image_B = Image.open(B_path)
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        # A是原图像，B为目标图像
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        A_outline = transforms.Grayscale(3)(item_A)
        B_outline = transforms.Grayscale(3)(item_B)

        return {"A": item_A, "B": item_B, "A_name": A_name, "B_name": B_name, "A_shape": A_shape, "A_outline": A_outline, "B_outline": B_outline}
        # return {"A": item_A, "A_name":A_name, "A_shape":A_shape}
        # return {"A": item_A, "A_name":A_name, "A_shape":A_shape, "A_outline": A_outline}

    def __len__(self):
        return len(self.files)



########################################################
# Replay Buffer
########################################################

class ReplayBuffer:
    # We keep an image buffer that stores
    # the 50 previously created images.
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # Returns newly added image with a probability of 0.5.
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element  # replaces the older image with the newly generated image.
                else:
                    # Otherwise, it sends an older generated image and
                    to_return.append(element)
        return Variable(torch.cat(to_return))


########################################################
# change lr according to the training epoch
########################################################


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
                       n_epochs - decay_start_epoch
               ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # Below line checks whether the current epoch has exceeded the decay epoch(which is 100)
        # e.g. if current epoch is 80 then max (0, 80 - 100) will be 0.
        # i.e. then entire numerator will be 0 - so 1 - 0 is 1
        # i.e. the original LR remains as it is.
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
                self.n_epochs - self.decay_start_epoch
        )


########################################################
# 初始化卷积层参数为 N(0,0.02)
########################################################


def initialize_conv_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


########################################################
# save and load checkpoints
########################################################

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


