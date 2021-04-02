import torch
import math


def xavier(m, h):
    # uniform 均匀分布
    return torch.Tensor(m, h).uniform_(-1, 1) * math.sqrt(6. / (m + h))


def relu(x):
    x[x < 0] = 0
    return x

def kaiming(m,h):
    return torch.randn(m,h)*math.sqrt(2./m)

if __name__ == '__main__':
    # # 输出爆炸
    # x = torch.randn(512)
    # for i in range(100):
    #     a = torch.randn(512, 512)
    #     x = a @ x
    # print(x.mean(), x.std())
    #
    # # 输出消失
    # x = torch.randn(512)
    # for i in range(100):
    #     a = torch.randn(512, 512) * 0.01
    #     x = a @ x
    # print(x.mean(), x.std())
    #
    # # y和 a x 的均值方差的关系
    # mean, var = 0., 0.
    # for i in range(10000):
    #     x = torch.randn(512)
    #     a = torch.randn(512, 512)
    #     y = a @ x
    #     mean += y.mean().item()
    #     var += y.pow(2).mean().item()
    # print(mean / 10000, math.sqrt(var / 10000))
    #
    # x = torch.randn(512)
    # for i in range(100):
    #     a = torch.randn(512, 512) * math.sqrt(1. / 512)
    #     x = a @ x
    # print(x.mean(), x.std())
    #
    # x = torch.randn(512)
    # for i in range(100):
    #     a = torch.randn(512,512)*math.sqrt(1./512)
    #     x = torch.tanh(a@x)
    # print(x.mean(), x.std())

    # x = torch.randn(512)
    # for i in range(100):
    #     a = torch.randn(512, 512).uniform_(-1, 1) * math.sqrt(1. / 512)
    #     x = torch.tanh(a @ x)
    # print(x.mean(), x.std())

    # x = torch.randn(512)
    # for i in range(100):
    #     a = xavier(512, 512)
    #     x = torch.tanh(a @ x)
    # print(x.mean(), x.std())

    # # y和 a x 的均值方差的关系
    # mean, var = 0., 0.
    # for i in range(10000):
    #     x = torch.randn(3,10)
    #     a = torch.randn(1, 3)
    #     y = relu(a @ x)
    #     mean += y.mean().item()
    #     var += y.pow(2).mean().item()
    # print(mean / 10000, math.sqrt(var / 10000))

    x = torch.randn(512)
    for i in range(100):
        a = kaiming(512, 512)
        x = relu(a @ x)
    print(x.mean(), x.std())


    # x = torch.randn(512)
    # for i in range(100):
    #     a = xavier(512, 512)
    #     x = relu(a @ x)
    # print(x.mean(), x.std())
