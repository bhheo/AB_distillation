'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from models import *


def criterion_alternative_L2(source, target, margin):
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Parameters
gpu_num = 0

temperature = 3
base_lr = 0.1
KD = True

distill_epoch = 1
max_epoch = 5

use_cuda = torch.cuda.is_available()

# Dataset load
transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113, 9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113, 9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

distillloader = trainloader


# Model
t_net = torch.load('./results/WRN22-4_200epoch_final.t7', map_location=lambda storage, location: storage)['net']
# # version issue
# for m in t_net.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
teacher = WRN22_4()
teacher.load_state_dict(t_net.state_dict())
student = WRN16_2()

s_net = Distill_WRN_Simple(student)
t_net = Distill_WRN_Simple(teacher)

# soft connection
d_net = Active_Soft_WRN_norelu(teacher, s_net)

if use_cuda:
    torch.cuda.set_device(gpu_num)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()
optimizer = optim.SGD(s_net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

# Training
def train_distillation(d_net, s_net, epoch):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)
    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(currentloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        batch_size = inputs.shape[0]
        outputs = d_net(inputs)

        # Activation transfer loss
        loss_AT1 = ((d_net.Connect1(d_net.res1) > 0) ^ (d_net.res1_t.detach() > 0)).sum().float() / d_net.res1_t.nelement()
        loss_AT2 = ((d_net.Connect2(d_net.res2) > 0) ^ (d_net.res2_t.detach() > 0)).sum().float() / d_net.res2_t.nelement()
        loss_AT3 = ((d_net.Connect3(d_net.res3) > 0) ^ (d_net.res3_t.detach() > 0)).sum().float() / d_net.res3_t.nelement()

        # Alternative loss
        margin = 1.0

        loss_alter = criterion_alternative_L2(d_net.Connect3(d_net.res3), d_net.res3_t.detach(), margin) / batch_size
        loss_alter += criterion_alternative_L2(d_net.Connect2(d_net.res2), d_net.res2_t.detach(), margin) / batch_size / 2
        loss_alter += criterion_alternative_L2(d_net.Connect1(d_net.res1), d_net.res1_t.detach(), margin) / batch_size / 4

        loss = loss_alter / 1000 * 3

        loss.backward()
        optimizer.step()

        train_loss1 += loss_AT1.item()
        train_loss2 += loss_AT2.item()
        train_loss3 += loss_AT3.item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('layer1_activation similarity %.1f%%' % (100 * (1 - train_loss1 / (b_idx+1))))
    print('layer2_activation similarity %.1f%%' % (100 * (1 - train_loss2 / (b_idx+1))))
    print('layer3_activation similarity %.1f%%' % (100 * (1 - train_loss3 / (b_idx+1))))

# Training
def train(net, epoch):
    epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(currentloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (b_idx + 1)

# Training
def train_KD(t_net, s_net, epoch):
    epoch_start_time = time.time()
    print('\nClassification training Epoch: %d' % epoch)
    s_net.train()
    t_net.eval()
    train_loss = 0
    correct = 0
    total = 0
    global optimizer
    for batch_idx, (inputs, targets) in enumerate(currentloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size1 = inputs.shape[0]
        batch_size2 = inputs.shape[0]

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out_t = t_net(inputs)
        outputs = s_net(inputs)

        loss_ori = criterion_CE(outputs[0:batch_size1, :], targets)
        loss_KD = - (F.softmax(t_net.out / temperature, 1).detach() *
                     (F.log_softmax(s_net.out / temperature, 1) - F.log_softmax(t_net.out / temperature, 1).detach())
                     ).sum() / batch_size2

        loss = loss_KD# + loss_ori

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs[0:batch_size1, :].data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (b_idx + 1)

def test(net, epoch, save=False):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total


# Distillation (Initialization)
currentloader = distillloader
for epoch in range(1, int(distill_epoch) + 1):
    if epoch == 1:
        optimizer = optim.SGD([{'params': s_net.parameters()},
                               {'params': d_net.Connect1.parameters()},
                               {'params': d_net.Connect2.parameters()},
                               {'params': d_net.Connect3.parameters()}], lr=base_lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    elif epoch == math.ceil(distill_epoch * 0.75) + 1:
        optimizer = optim.SGD([{'params': s_net.parameters()},
                               {'params': d_net.Connect1.parameters()},
                               {'params': d_net.Connect2.parameters()},
                               {'params': d_net.Connect3.parameters()}], lr=base_lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
    train_distillation(d_net, s_net, epoch)

# Cross-entropy training
currentloader = trainloader
final_acc = AverageMeter()
optimizer = optim.SGD(s_net.parameters(), lr=base_lr, nesterov=True, momentum=0.9, weight_decay=5e-4)
for epoch in range(1, max_epoch+1):
    if epoch == math.ceil(max_epoch*0.3)+1:
        optimizer = optim.SGD(s_net.parameters(), lr=base_lr/5, nesterov=True, momentum=0.9, weight_decay=5e-4)
    elif epoch == math.ceil(max_epoch*0.6)+1:
        optimizer = optim.SGD(s_net.parameters(), lr=base_lr/(5*5), nesterov=True, momentum=0.9, weight_decay=5e-4)
    elif epoch == math.ceil(max_epoch*0.8)+1:
        optimizer = optim.SGD(s_net.parameters(), lr=base_lr/(5*5*5), nesterov=True, momentum=0.9, weight_decay=5e-4)

    if KD is True:
        train_loss = train_KD(t_net, s_net, epoch)
    else:
        train_loss = train(s_net, epoch)

    test_loss, accuracy = test(s_net, epoch, save=True)
    if epoch > max_epoch - max_epoch / 20:
        final_acc.update(accuracy)

final_acc = final_acc.avg

print('\nFinal 10 average Acc: %.3f%%' % (100 * final_acc))

# state = {
#     'net': s_net,
#     'epoch': max_epoch,
# }
# torch.save(state, './results/%depoch_final.t7' % (max_epoch))