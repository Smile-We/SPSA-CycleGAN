from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from sa_cyclegan import GeneratorResNet
from utils import ImageDataset
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    dataset_train_mode="train",
    dataset_test_mode="test",
    img_size=256,
    channels=3,
    num_residual_blocks=9,
)

def stain_norm(input_direc, output_direc, param_direc, mode):
    transforms_ = [
        transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    img_dataloader = DataLoader(
        ImageDataset(input_direc, mode=hp.dataset_train_mode, transforms_=transforms_),
        batch_size=6,
        shuffle=True,
        num_workers=8,
    )
    input_shape = (hp.channels, hp.img_size, hp.img_size)

    Generator = GeneratorResNet(input_shape, hp.num_residual_blocks, mode).to(device)
    Generator.load_state_dict(torch.load(param_direc+"gen_199_a.pth.tar", map_location=device)["state_dict"], strict=False)
    for param in Generator.parameters():
        param.requires_grad = False
    Generator.eval()

    for batch in img_dataloader:
        real = Variable(batch["A"].float()).to(device)
        name = batch["A_name"]
        origin_shape = batch["A_shape"]
        fake = Generator(real)
        for i in range(fake.shape[0]):
            shape = origin_shape[i]
       
            img_grid = make_grid(fake[i], nrow=1, normalize=True)
            img = transforms.ToPILImage()(img_grid)
            img.save(output_direc+f"gen_{name[i]}")


if __name__ == "__main__":
    root = "/home/SPSA_CycleGAN/"

    img_direc = "data/test/processed_img/"
    param_direc = "param/none/"
    model_mode = ['att_plus', 'no_att']
    output_direc = root+"data/test/output/none/"

    device = "cuda"
    stain_norm(root+img_direc, output_direc, root+param_direc, model_mode[1])
