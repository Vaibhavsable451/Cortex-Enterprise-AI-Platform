"""BatchNorm and LayerNorm implemented by hand."""
import torch
import torch.nn as nn


class MyBatchNorm1d(nn.Module):
    def __init__(self, n, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.m = eps, momentum
        self.gamma, self.beta = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))
        self.register_buffer("rm", torch.zeros(n))
        self.register_buffer("rv", torch.ones(n))

    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                self.rm.lerp_(mu, self.m)
                self.rv.lerp_(var, self.m)
        else:
            mu, var = self.rm, self.rv
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


class MyLayerNorm(nn.Module):
    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma, self.beta = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))

    def forward(self, x):
        mu, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta


if __name__ == "__main__":
    x = torch.randn(16, 8)
    assert torch.allclose(MyBatchNorm1d(8)(x), nn.BatchNorm1d(8)(x), atol=1e-5)
    assert torch.allclose(MyLayerNorm(8)(x), nn.LayerNorm(8)(x), atol=1e-5)
    print("BatchNorm / LayerNorm match PyTorch ✔")
