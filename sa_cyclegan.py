import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nystrom_attention import NystromAttention


##############################################
# 两层卷积的残差网络
##############################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, mode, pe=None, num_hidden=4*4, head=8, sample_ratio=4):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.mode = mode
        # self.pe = pe
        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ),  # Reflection padding is used because it gives better image quality at edges.
            nn.Conv2d(
                in_channel, in_channel, 3
            ),  # Paper says - same number of filters on both layer.
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        )
        self.down_sample = []
        for _ in range(sample_ratio):
            self.down_sample += [
                nn.Conv2d(
                    self.in_channel, self.in_channel, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            ]
        self.down_sample = nn.Sequential(*self.down_sample)

        self.up_sample = []
        for _ in range(sample_ratio):
            self.up_sample += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    self.in_channel, self.in_channel, kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            ]
        self.up_sample = nn.Sequential(*self.up_sample)

        self.attn = NystromAttention(
            dim=num_hidden,
            dim_head=num_hidden//head, #   new att 64
            heads=head,
            num_landmarks=6,  # number of landmarks             new att 256
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        '''
        这里同样使用上下采样的方式对x进行缩放
        '''
        # output = self.pooling(x)
        
        # return x + self.block(x)+output
        if self.mode == 'no_att':
            return x + self.block(x)
        else:
            output = self.down_sample(x)
            shape = output.shape
            output = output.reshape(shape[0], shape[1], -1)
            # output = output + self.pe
            output = self.attn(output)
            output = output.reshape(shape[0], shape[1], shape[2], shape[3])
            output = self.up_sample(output)
            return x + self.block(x) + output


##############################################
# 生成器
##############################################


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, mode):
        super(GeneratorResNet, self).__init__()

        self.mode = mode
        channels = input_shape[0]

        # Initial convolution block
        out_channels = 64
        # I define a variable 'model' which I will continue to update
        # throughout the 3 blocks of Residual -> Downsampling -> Upsampling
        # First c7s1-64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels

        # 降采样
        # d128 => d256
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        """ 残差网络
        R256,R256,R256,R256,R256,R256,R256,R256,R256
        """
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels, self.mode)]

        # 上采样
        # u128 => u64
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # 输出层
        # c7s1-3
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

        # '''
        # 应该使用上采样的方式使矩阵的值平稳回复到图像的大小
        # '''
        # self.up_layer = nn.AdaptiveAvgPool2d((256, 256))
        # self.down_layer = nn.Conv2d(6,3,1,1,0)
        # self.down_layer = nn.Conv2d(4,3,1,1,0) #这里是修改了w添加方式

    # y是4*3*2*1的矩阵
    def forward(self, x, y=None):
        if y is not None:
            y = self.up_layer(y)
            y = torch.cat((y,y,y),1) # 若修改w添加方式之后则不用
            z = self.down_layer(torch.cat((x,y),1))
        else:
            z = x
        return self.model(z)


##############################
# 判别器
##############################
""" 使用70x70的patch"""


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # C64 -> C128 -> C256 -> C512
        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(64, out_channels=128),
            *discriminator_block(128, out_channels=256),
            *discriminator_block(256, out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    b = torch.zeros((4, 3, 256, 256))
    gen = GeneratorResNet((3, 256, 256), 9, "att")
    gen_b = gen(b)
    print(gen_b.shape)
