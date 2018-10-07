import model

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import argparse


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train(data_path="", gpu=-1, load_model=None, save_model=None):
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # data
    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)

    # model
    net = model.ImageNet(path=load_model)
    if gpu >= 0:
        net.cuda()

    # train
    print('Start Training')
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(10):
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            # 输入数据
            if gpu >= 0:
                x, y = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                x, y = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            # 更新参数
            optimizer.step()

            # 打印log
            running_loss += loss.data[0]
            if (step + 1) % 100 == 0:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
    print('Finish Training')

    print('Start Testing')
    net.eval()
    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=1)
    test_loss = 0.0
    correct = 0
    total = 0
    for step, (inputs, labels) in enumerate(test_loader):
        # 输入数据
        if gpu >= 0:
            x, y = Variable(inputs, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()
        else:
            x, y = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        outputs = net(x)
        loss = criterion(outputs, y)

        test_loss += loss.data[0]
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().data[0]
    print('loss: %.3f | accuracy: %.3f %%' % (test_loss / total, 100. * correct / total))
    print('Finish Testing')

    net.cpu()
    if save_model is not None:
        torch.save(net.state_dict(), './models/net_params.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--gpu', type=int, default=9)
    parser.add_argument('--data', type=str, default="./")
    parser.add_argument('--load_model', type=str, default="./models/net_params_pretrain.pkl")
    parser.add_argument('--save_model', type=str, default="./models/net_params.pkl")
    args = parser.parse_args()

    train(data_path=args.data, gpu=args.gpu, load_model=args.load_model, save_model=args.save_model)

