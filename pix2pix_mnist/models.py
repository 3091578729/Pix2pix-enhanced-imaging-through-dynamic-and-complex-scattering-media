import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
       
        x = self.model(x)

        x = torch.cat((x, skip_input), 1)

        return x
    

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
    



#####################################################
#             UNet with Resblock             
#####################################################

class ResUNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResUNetDown, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, 3, 2, 1)
    #    self.bn1 = nn.InstanceNorm2d(out_size)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)
    #    self.bn2 = nn.InstanceNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(in_size, out_size, 3, 2 ,1)

        self.relu4 = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
       # y = self.conv2(y)
       # y = self.bn2(y)
       # y = self.relu2(y)

        x = self.conv3(x)

        out = x + y
        out = self.relu4(out)
        return out


class ResUNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResUNetUp, self).__init__()

        res_out_size = out_size // 2
        self.upconv1 = nn.ConvTranspose2d(in_size, in_size, 2, 2)

        self.conv1 = nn.Conv2d(in_size, res_out_size, 3, 1, 1)
    #    self.bn1 = nn.InstanceNorm2d(out_size)
        self.bn1 = nn.BatchNorm2d(res_out_size)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(res_out_size, res_out_size, 3, 1, 1)
    #    self.bn2 = nn.InstanceNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(res_out_size)
        self.relu2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(in_size, res_out_size, 3, 1 ,1)

        self.relu4 = nn.LeakyReLU(0.2)
        

    def forward(self, x, skip_input):

        y1 = self.upconv1(x)
        
        y2 = self.conv1(y1)
        y2 = self.bn1(y2)
        y2 = self.relu1(y2)

        y2 = self.conv2(y2)
        y2 = self.bn2(y2)
        y2 = self.relu2(y2)

        y3 = self.conv3(y1)

        out = y2 + y3
        out = self.relu4(out)

        out = torch.cat((out, skip_input), 1)
   
        return out


class GeneratorResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorResUNet, self).__init__()
        '''
        self.down1 = ResUNetDown(in_channels, 64)
        self.down2 = ResUNetDown(64, 128)
        self.down3 = ResUNetDown(128, 256)
        self.down4 = ResUNetDown(256, 512)
        self.down5 = ResUNetDown(512, 512)
        self.down6 = ResUNetDown(512, 512)
        self.down7 = ResUNetDown(512, 512)
        self.down8 = ResUNetDown(512, 512)

        self.up1 = ResUNetUp(512, 512*2)
        self.up2 = ResUNetUp(1024, 512*2)
        self.up3 = ResUNetUp(1024, 512*2)
        self.up4 = ResUNetUp(1024, 512*2)
        self.up5 = ResUNetUp(1024, 256*2)
        self.up6 = ResUNetUp(512, 128*2)
        self.up7 = ResUNetUp(256, 64*2)
        '''
        self.down1 = ResUNetDown(in_channels, 32)
        self.down2 = ResUNetDown(32, 64)
        self.down3 = ResUNetDown(64, 128)
        self.down4 = ResUNetDown(128, 256)
        self.down5 = ResUNetDown(256, 256)
        self.down6 = ResUNetDown(256, 256)
        self.down7 = ResUNetDown(256, 256)
        self.down8 = ResUNetDown(256, 256)

        self.up1 = ResUNetUp(256, 256*2)
        self.up2 = ResUNetUp(512, 256*2)
        self.up3 = ResUNetUp(512, 256*2)
        self.up4 = ResUNetUp(512, 256*2)
        self.up5 = ResUNetUp(512, 128*2)
        self.up6 = ResUNetUp(256, 64*2)
        self.up7 = ResUNetUp(128, 32*2)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
        #    nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh(),
        )



    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)



##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
