import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

import argparse
parser = argparse.ArgumentParser(description='CapsNet with MNIST')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 250)')
parser.add_argument('--n_classes', type=int, default=10, metavar='N',
                    help='number of classes (default: 10)')

# if you want change the value                                               
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    # 3e-4 로도 바꿔보기
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--routing_iterations', type=int, default=3)
parser.add_argument('--with_reconstruction', action='store_true', default=False)
# reconstruction은 안쓸거니깐 그냥 false 그대로 놔도 될듯. ㅎ
args = parser.parse_args()

n_classes = args.n_classes


def squash(x):
    # print(f'original    x shape:{x.size()}')
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    # print(f'weight vector shape:{(lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1).size()}')
    # print(f'after squashX shape:{x.size()}')
    return x

class AgreementRouting(nn.Module):
    # num_primaryCaps(input_caps ) = 32 * 6 * 6  -> 1152
    # n_classes      (output_caps) = 10
    # n_iterations                 = 3  -> parser 에 나와있음.

    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))


    def forward(self, u_predict):
        # torch.Size([batch_size, 1152, 10, 16])
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=1)
        # dim 에러로 수정됨
        # torch.Size([1152, 10])
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        # [1152, 10, 1] * [batch_size, 1152, 10, 16]
        # [2, 1152, 10, 16] -> [2, 10, 16] batch만 빼고 계산 되네!
        # 10x16 의 행렬이 1152개 있다가 -> 10x16만 남음 다 더해졌네.
        v = squash(s)
        # relu 하는거라고 생각하면됨 (비선형성 추가)  [batch_size, 10, 16]

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            # [1152, 10] -> [batch_size, 1152, 10]
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                # [batch_size, 1, 10, 16]
                b_batch = b_batch + (u_predict * v).sum(-1)
                #  ([batch_size, 1152, 10, 16] * [batch_size, 1, 10, 16]).sum(-1)
                # ->[batch_size, 1152, 10]
                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                # dim 에러로 수정됨
                # c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                # F.softmax(b_batch.view(-1, output_caps), dim=1).size() -> [147456, 10]
                # .view(-1, input_caps, output_caps, 1) -> 

                s = (c * u_predict).sum(dim=1)
                v = squash(s)
                # [batch_size, 10, 16]
        return v

class CapsLayer(nn.Module):
    # digitcaps를 계산하는 부분이라고 봐도무방
    # input_caps = 32 * 6 * 6, input_dim = 8, output_caps = 10, output_dim = 16, routing_module
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        # self.weights -> 1152, 8, 160
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        # caps_output -> primary caps거친후의 input 말함
        # caps_output.size() => batch_size, 32x6x6, 8
        caps_output = caps_output.unsqueeze(2)
        # caps_output.size() => batch_size, 32x6x6(1152), 1, 8
        u_predict = caps_output.matmul(self.weights)
        # batch_size, 1152, 1, 8  matmul 1152, 8, 160
        # torch.Size([batch_size, 1152, 1, 160])
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        # torch.Size([batch_size, 1152, 10, 16])
        v = self.routing_module(u_predict)
        # torch.Size([batch_size, 10, 16])
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)

        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)
        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        # 여기서 비선형성이 추가되어져나온다. squash 라는게 그냥 비선형성을 위한 relu 같은 존재로 보면될듯
        # 마지막벡터에는 곱해지지않네요 print 찍어보면 암
        # print(f'output_size입니다: {out.size()}') -> 여튼 input vector랑, output vector랑 shape는 동일함.

        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=n_classes):
        # 
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)

        # 라우팅이 있어야 digitCaps를 계산할 수 있네요
        # 근데 agreementRouting 에서 forward 부분은 capsLayer에서 진행되기 때문에 CapsLayer를 먼저 보도록 하겠습니다.

        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        # [batch_size, 10, 16]
        probs = x.pow(2).sum(dim=2).sqrt()
        # [batch_size, 10]
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=n_classes):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs

class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        # (m_pos=0.9, m_neg=0.1, lambda_=0.5)
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        # lengths -> pred,   targets -> ground truth
        t = torch.zeros(lengths.size()).long()
        # print(t.size())
        # [batch_size, 10]
        # [batch_size, output]
        # print(targets.size())
        # [128]
        if targets.is_cuda:
            t = t.cuda()

        targets = t.scatter_(1, targets.data.view(-1, 1), 1)
        # t -> [batch_size, 10] 으로 이루어진 정답라벨임을 알 수 있고 그 정답라벨에 targets의 값들을 정답값으로 one hot encoding을 진행함을 알 수 있다.
        # targets -> [batch_size]로 이루어진 텐서이며 개개의 값은 0~9의 정답 라벨을 가진다.

        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 (1. - targets.float()) * self.lambda_ * F.relu(lengths - self.m_neg).pow(2)
        # ground truth * ReLU(0.9-pred) + (1-ground truth)*0.5*ReLU(pred-0.1)^2
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':

    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = CapsNet(args.routing_iterations)
    # routing_iterations == 3 이네요 ㅎ

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, 10)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)
    # reconstruction은 안쓸거니깐 그냥 false 그대로 놔도 될듯. ㅎ    

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target
            # data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()

            # print(data.size())
            # [batch_size, 1, 28, 28]
            # 채널이 1개인것도 염두해두어야함.

            # reconstruction은 없어서 else로 넘어감
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss

            else:
                output, probs = model(data)
                # print(output.size())
                # [128, 10, 16]
                # print(probs.size())
                # [128, 10]
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss.data))

    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                if args.with_reconstruction:
                    output, probs = model(data, target)
                    reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).data[0]
                    test_loss += loss_fn(probs, target, size_average=False).item()
                    test_loss += reconstruction_alpha * reconstruction_loss

                else:
                    output, probs = model(data)
                    test_loss += loss_fn(probs, target, size_average=False).item()
                    # loss_fn -> margin_loss


                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                #     probs:tensor([[0.0259, 0.0423, 0.0937,  ..., 0.0304, 0.0151, 0.0750],
                #     [0.8676, 0.0431, 0.0692,  ..., 0.0208, 0.0116, 0.0254]......

                # pred:tensor([[3],
                # [0],
                # [1], ....
                # 이중에 최고의 probability를 뽑는것
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                # pytorch에서 지원하는 기능 eq -> equal로서  pred와 target이 같은비율을 보려고 한거네요.

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            return test_loss

    args.epochs = 10
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
        torch.save(model.state_dict(),
                   '{:03d}_model_dict_{}routing_reconstruction{}.pt'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction))
