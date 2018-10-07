import torch
import torch.nn as nn
import torchvision.models as Models
from torch.autograd import Variable


class ImageNet(nn.Module):
    def __init__(self, path=None, c=10):
        super(ImageNet, self).__init__()
        self.feature = Models.resnet18(pretrained=True)
        self.feature = torch.nn.Sequential(*list(self.feature.children())[:-2])
        self.classifier = nn.Linear(512, c)
        # torch.save(self.feature.state_dict(), './models/net_params_pretrain.pkl')
        if path is not None:
            self.feature.load_state_dict(torch.load(path))

    def forward(self, x):
        x = self.feature(x)
        y = self.classifier(x.view(-1, 512))
        return y

if __name__ == '__main__':
    pass
    """
    print(Models.vgg19())
    x = Variable(torch.rand((1, 3, 32, 32)))
    M = ImageNet("./models/net_params_pretrain.pkl")
    print(M(x))
    """