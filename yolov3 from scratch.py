import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
import torch.optim as optim
# B - residual layer, ["B", 1] - 1 is the number of repeats
# S - scale prediction block and computing YOLO loss
# U - upsampling the feature map and concatnating with a prev layer
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8], 
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self,inchannels,outchannels,bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inchannels,outchannels,bias = not bn_act, **kwargs)
        self.batchnorm = nn.BatchNorm2d(outchannels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forwarf(self,x):
        if self.use_bn_act:
            return self.leaky(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)


class Residualblock(nn.Module):
    def __init__(self, channels, use_residual = True, numrepeat = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(numrepeat):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels,channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size = 3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.numrepeat = numrepeat

        def forward(self,x):
            for layer in self.layers:
                if self.use_residual:
                    x= layer(x) + x
                else:
                    x= layer(x)
            return x
        
class Scaleprediction(nn.Module):
    def __init__(self,inchannels, numclasses):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(inchannels, 2*inchannels, kernel_size = 3, padding=1),
            CNNBlock(2*inchannels, (numclasses+5)*3,bn_act=False, kernel_size = 1), #here, the 5 indicates 5 values such as object's probability,x,y,w,h
        )
        self.numclasses = numclasses
    
    def forward(self,x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.numclasses + 5, x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)
        )


class Yolov3(nn.Module):
    def __init__(self, inchannels = 3, numclasses = 20 ):
        super().__init__()
        self.numclasses = numclasses
        self.inchannels = inchannels
        self.layers = self.createconvlayers()
    
    def forward (self,x):
        outputs = []
        routeconnections = []

        for layer in self.layers:
            if isinstance(layer, Scaleprediction):
                outputs.append(layer(x))
                continue

            x=layer(x)

            if isinstance(layer, Residualblock) and layer.numrepeat == 8:
                routeconnections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, routeconnections[-1]], dim=1 )
                routeconnections.pop()
        return outputs

    def createconvlayers(self):
        layers = nn.ModuleList()
        inchannels = self.inchannels

        for module in config:
            if isinstance(module, tuple):
                outchannels, kernel_size, stride = module
                layers.append(CNNBlock(
                    inchannels,outchannels,kernel_size=kernel_size,stride=stride,padding=1 if kernel_size == 3 else 0,
                ))

                inchannels = outchannels

            elif isinstance(module, list):
                numrepeats = module[1]
                layers.append(Residualblock(inchannels, numrepeat=numrepeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        Residualblock(inchannels, use_residual=False, numrepeat=1),
                        CNNBlock(inchannels,inchannels//2,kernel_size=1),
                        Scaleprediction(inchannels//2, numclasses=self.numclasses)
                    ]
                    inchannels = inchannels//2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    inchannels = inchannels * 3

        return layers   

if __name__ == "__main__":
    numclasses = 20
    image_size = 416
    model = Yolov3(numclasses=numclasses)
    x = torch.randn((2,3,image_size,image_size))
    out = model(x)
    assert model(x)[0].shape == (2,3,image_size//32,image_size//32,numclasses+5)
    assert model(x)[1].shape == (2,3,image_size//16,image_size//16,numclasses+5)
    assert model(x)[2].shape == (2,3,image_size//8,image_size//8,numclasses+5)
    print("ye")