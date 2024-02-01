# 作者：冰雪

import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
# print(input)
# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("../logs/logs_sigmoid")

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return  output

tudui = Tudui()
# output = tudui(input)
# print(output)

for idx, data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images("input", imgs, idx)
    output = tudui(imgs)
    writer.add_images("output", output, idx)

writer.close()