import torch
from torch import tensor

def quantGrad():
    input = torch.randn(4, requires_grad=True)
    output = torch.sign(input)
    loss = output.mean()
    loss.backward()
    # 梯度为0的原因，sign函数导数为0
    print(input.grad)

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # 由阶跃变成平滑过渡
        return grad_output.clamp_(-1, 1)

def fixQuantGrad():
    sign = LBSign.apply
    params = torch.randn(4, requires_grad = True)
    output = sign(params)
    # mean的梯度为1/size
    loss = output.mean()
    loss.backward()
    print(params.grad)

if __name__ == '__main__':
    pass
