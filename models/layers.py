import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if p is None:
        p = k // 2 * d
    return p

class Conv(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))#x经过cv1后分成2个通道
        y.extend(m(y[-1]) for m in self.m)#每个通道分别经过n个bottleneck，每个输出都记录在y中，y=（特征块1，特征块2，bottleneck1...）
        return self.cv2(torch.cat(y, 1))#返回y里面数的相加

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:#与chunk不同，split是直接分成2个通道，而chunk是分成2个张量
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))#getattr(self, "n", 3) 安全地获取 self.n 的值，若不存在则使用默认值 3
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):#'nearest'（最近邻插值）或 'linear'（线性插值）
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        return self.upsample(x)

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension
    
    def forward(self, x):
        return torch.cat(x, dim=self.dimension)#按维度进行拼接

class DFL(nn.Module):#对输出的每个通道进行softmax，然后对每个通道进行加权求和，得到最终的输出
    def __init__(self, c1=16):
        super().__init__()
        self.reg_max = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
    
    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        # 确保输入通道数是 4 * reg_max
        assert c == 4 * self.reg_max, f"Expected input channels to be 4 * reg_max ({4 * self.reg_max}), got {c}"
        # 重塑输入为 [batch, 4, reg_max, anchors]
        x = x.view(b, 4, self.reg_max, a)
        # 对每个坐标的 reg_max 个通道进行处理
        x = x.permute(0, 2, 1, 3).reshape(b, self.reg_max, -1, 1)
        # 应用卷积并重塑输出
        x = self.conv(x).view(b, 4, a)
        return x