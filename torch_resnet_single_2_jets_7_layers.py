import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable


# initialize the resblock with three layers. 
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)
        self.debug = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.debug: print(out.size(), 'Layer Conv1')
        out = self.relu(out)
        if self.debug: print(out.size(), 'Layer Relu')
        out = self.conv2(out)
        if self.debug: print(out.size(), 'Layer Conv2')

        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out
    

# initialize the transpose resblock with three layers. 
class ResBlockTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, out_shape=None):
        super(ResBlockTranspose, self).__init__()
        self.upsample = in_channels//out_channels
        self.conv1_tr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=self.upsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_tr = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut_tr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=self.upsample)
        self.out_shape = out_shape
        self.debug = False

    def forward(self, x):
        residual = x
        if self.out_shape is not None:
            out = self.conv1_tr(x, output_size=(x.size()[0], x.size()[1], self.out_shape[0], self.out_shape[1]))
        else:
            out = self.conv1_tr(x)
        if self.debug: print(out.size(), 'Layer Conv1')
        out = self.relu(out)
        if self.debug: print(out.size(), 'Layer Relu')
        out = self.conv2_tr(out)
        if self.debug: print(out.size(), 'Layer Conv2')

        if self.upsample > 1:
            if self.out_shape is not None:
                residual = self.shortcut_tr(x, output_size=(x.size()[0], x.size()[1], self.out_shape[0], self.out_shape[1]))
            else:
                residual = self.shortcut_tr(x)

        out += residual
        out = self.relu(out)

        return out


# the encoder. it uses 7 resnet blocks in its structure.
class Encoder(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(Encoder, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        
        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=7, stride=2, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.layer4 = self.block_layers(1, [fmaps[1],fmaps[2]])
        self.layer5 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer6 = self.block_layers(1, [fmaps[2],fmaps[3]])
        self.layer7 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])

        self.fc = nn.Linear(fmaps[1], 1)
        self.debug = False


        
    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.debug: print(x.size(), 'Input')
        x = self.conv0(x)
        if self.debug: print(x.size(), 'Conv0')
        x = F.relu(x)
        if self.debug: print(x.size(), 'Relu')
        x = F.max_pool2d(x, kernel_size=2)
        if self.debug: print(x.size(), 'Maxpooling')
        x = self.layer1(x)
        if self.debug: print(x.size(), 'Layer1')
        x = self.layer2(x)
        if self.debug: print(x.size(), 'Layer2')
        x = self.layer3(x)
        if self.debug: print(x.size(), 'Layer3')    
        x = self.layer4(x)
        if self.debug: print(x.size(), 'Layer4')
        x = self.layer5(x)
        if self.debug: print(x.size(), 'Layer5')    
        x = self.layer6(x)
        if self.debug: print(x.size(), 'Layer6')
        x = self.layer7(x)
        if self.debug: print(x.size(), 'Layer7')    


        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        if self.debug: print(x.size())        
        x = x.view(x.size()[0], -1) #self.fmaps[1])
        if self.debug: print(x.size())
        # x = self.fc(x)
        #x = self.FCN(x)
        return x


# the decoder. it uses 7 resnet blocks in its structure.
class Decoder(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(Decoder, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.in_channels = in_channels
        self.conv01 = nn.ConvTranspose2d(fmaps[0], fmaps[0],  kernel_size=2, stride=1, padding=0)
        self.conv02 = nn.ConvTranspose2d(fmaps[0], in_channels,  kernel_size=5, stride=2, padding=0)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[3],fmaps[3]])
        self.layer2 = self.block_layers(1, [fmaps[3],fmaps[2]], out_shape=(5,5))
        self.layer3 = self.block_layers(self.nblocks, [fmaps[2],fmaps[2]])
        self.layer4 = self.block_layers(1, [fmaps[2],fmaps[1]], out_shape=(10,10))
        self.layer5 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])

        self.layer6 = self.block_layers(1, [fmaps[1],fmaps[0]], out_shape=(20,20))
        self.layer7 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])

        self.fc = nn.Linear(self.fmaps[-1], self.fmaps[-1]*3*3)
        self.debug = False

    def block_layers(self, nblocks, fmaps, out_shape=None):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlockTranspose(fmaps[0], fmaps[1], out_shape))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        if self.debug: print(x.size(), 'Decoder Input')
        x = x.view(-1, self.fmaps[-1], 3, 3) # 3x3, down4/5
        if self.debug: print(x.size())        

        x = self.layer1(x)
        if self.debug: print(x.size(), 'Decoder Layer1')
        x = self.layer2(x)
        if self.debug: print(x.size(), 'Decoder Layer2')
        x = self.layer3(x)
        if self.debug: print(x.size(), 'Decoder Layer3') 
        x = self.layer4(x)
        if self.debug: print(x.size(), 'Decoder Layer4')
        x = self.layer5(x)
        if self.debug: print(x.size(), 'Decoder Layer5') 
        x = self.layer6(x)
        if self.debug: print(x.size(), 'Decoder Layer6')
        x = self.layer7(x)
        if self.debug: print(x.size(), 'Decoder Layer7') 

        x = F.interpolate(x, scale_factor=2)
        if self.debug: print(x.size(), 'Decoder Interpolation')        
        x = self.conv01(x)
        if self.debug: print(x.size(), 'Decoder Conv01')
        x = F.relu(x, inplace = True)
        if self.debug: print(x.size(), 'Decoder Relu')

        x = self.conv02(x)
        if self.debug: print(x.size(), 'Decoder Conv02')
        x = F.relu(x, inplace = True)
        if self.debug: print(x.size(), 'Decoder Relu')
        return x

class Binary(Function):
    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# the autoencoder
class AutoEncoder(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps):
        super(AutoEncoder, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.encoder = Encoder(in_channels, self.nblocks, self.fmaps)
        self.binary = Binary()
        self.decoder = Decoder(in_channels, self.nblocks, self.fmaps)
        self.debug = False

    def forward(self,x):
        x = self.encoder(x)
        x = self.binary.apply(x)
        x = self.decoder(x)
        return x
