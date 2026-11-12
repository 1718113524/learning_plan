# 教程 02:PyTorch 训练循环(W3-4,你的最大短板)

> 对应计划 Phase 2。目标:2 周内**完全自己写**一个可用的训练流程。
> 学完的标志(M2):CIFAR-10 用你自己的训练循环达到 90%+ 准确率。

---

## 1. Tensor 速查(30 分钟过一遍)

| 操作 | 例子 | 说明 |
|---|---|---|
| 创建 | `torch.zeros(2,3)` / `randn` / `arange` | |
| 索引 | `x[0]`、`x[:, 1:]`、布尔索引 `x[x>0]` | |
| 形状 | `view` / `reshape` / `squeeze` / `unsqueeze` | view 要求内存连续,reshape 更宽松 |
| 转置 | `permute(0,2,1)` / `transpose(-1,-2)` | **permute 后内存不连续,接 view 前先 .contiguous()** |
| 广播 | 形状 (2,1)+(3) → (2,3) | 从右往左对齐,维度为 1 或缺失才能广播 |
| 设备 | `x.to("cuda")` / `x.cuda()` / `x.cpu()` | 模型和数据必须在同一设备 |
| 精度 | `x.half()` / `x.bfloat16()` / `x.float()` | bf16 训练,float32 算指标 |

## 2. autograd 四问四答(面试也考)

**Q1:`loss.backward()` 到底干了什么?**
沿着计算图反向传播,把梯度**累加**到每个叶子张量的 `.grad` 里。所以每次迭代前必须 `optimizer.zero_grad()`,否则梯度越积越大。

**Q2:`.grad` 存在哪?**
只存在 `requires_grad=True` 的叶子张量上(模型的 `nn.Parameter` 就是叶子)。中间变量用完即弃,这就是显存里"激活"的来源。

**Q3:`detach()` 是什么?**
返回一个共享数据但**断开计算图**的新张量。典型场景:生成时把上一步输出转成下一步输入,不让梯度回传;记录 loss 数值时用 `.item()` 把 GPU 张量变成 Python 浮点(自带 detach)。

**Q4:`no_grad()` 和 `inference_mode()` 的区别?**
都禁用梯度计算、省显存。区别:`inference_mode` 更快,但其中的张量之后不能再用于需要梯度的计算;`no_grad` 更宽容。**评估循环用 `inference_mode` 就够了。**

## 3. Dataset / DataLoader(必背模板)

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):            # 必须实现
        return len(self.texts)

    def __getitem__(self, i):     # 必须实现:返回一条样本
        return self.texts[i], self.labels[i]

loader = DataLoader(
    MyDataset(texts, labels),
    batch_size=64,
    shuffle=True,        # 训练集打乱,验证集 False
    num_workers=2,       # 加载进程数,Windows 见下方坑
    pin_memory=True,     # GPU 训练时开,加速 H2D 拷贝
    drop_last=False,     # 最后一个不满 batch 是否丢弃
)
```

### ⚠️ Windows 大坑:num_workers

- `num_workers > 0` 时,DataLoader 用多进程加载,**训练入口必须包在 `if __name__ == "__main__":` 里**,否则报 "An attempt has been made to start a new process" 错误。
- 第一次跑建议 `num_workers=0` 调通,再改 2 对比速度。

## 4. ✅ M2 项目:CIFAR-10 完整训练脚本

**先装依赖:**
```bash
pip install torch torchvision tensorboard torchinfo
```

**train_cifar.py**(每个环节都有注释,建议逐段抄进 VS Code 跑)

```python
"""CIFAR-10 从零手写训练循环(不用任何 Trainer)。

用法:
    python train_cifar.py --epochs 50 --batch-size 128 --lr 0.01
    tensorboard --logdir runs      # 另开终端看曲线
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------- 模型:一个 6 层 BasicBlock 的小 ResNet ----------
class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:   # 尺寸不匹配才需要 1x1 卷积
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


# ---------- 数据 ----------
def get_dataloaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),      # 数据增强:随机裁剪+翻转防过拟合
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR-10 标准均值方差
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    val_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=100, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ---------- 评估 ----------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()                     # 关掉 BN/Dropout 的训练行为
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


# ---------- 训练一个 epoch ----------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer, use_amp, scaler):
    model.train()
    running_loss = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)          # set_to_none 比填 0 省一点显存
        if use_amp:                                    # 混合精度:前向 bf16,梯度回传后用 scaler 修正
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                 # 裁剪前必须先 unscale
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:                                          # CPU 或无 AMP 的朴素路径
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        running_loss += loss.item()
        if step % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(loader) + step)
    return running_loss / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 建议先 0,调通后再试 2")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device = {device}, amp = {use_amp}")

    train_loader, val_loader = get_dataloaders(args.batch_size, args.num_workers)

    model = ResNet().to(device)
    # 打印结构(可选,面试时也会被要求描述模型):
    # from torchinfo import summary; summary(model, input_size=(1, 3, 32, 32))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Warmup(前 2 epoch 线性升温) + Cosine 衰减
    def lr_lambda(epoch: int) -> float:
        if epoch < 2:
            return (epoch + 1) / 2
        progress = (epoch - 2) / max(1, args.epochs - 2)
        return 0.5 * (1 + np.cos(np.pi * progress))    # 1 → 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch, best_acc = 0, 0.0
    ckpt_dir = Path("./ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    if args.resume:                                    # 断点续训
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]

    writer = SummaryWriter("runs/cifar")               # 另开终端: tensorboard --logdir runs

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, writer, use_amp, scaler)
        acc = evaluate(model, val_loader, device)
        scheduler.step()                               # 每个 epoch 调一次学习率
        lr_now = optimizer.param_groups[0]["lr"]
        writer.add_scalar("val/acc", acc, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)
        print(f"epoch {epoch:3d} | loss {train_loss:.3f} | val acc {acc:.2%} | lr {lr_now:.2e}")
        if acc > best_acc:                             # 保存验证集最优模型
            best_acc = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_acc": best_acc},
                       ckpt_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_acc": best_acc}, ckpt_dir / "last.pt")

    print(f"best val acc = {best_acc:.2%}")


if __name__ == "__main__":     # Windows + num_workers>0 必须要有这个保护
    main()
```

**跑起来(CPU 也可以,只是慢):**
```bash
python train_cifar.py --epochs 50
# 另开一个终端看曲线:
tensorboard --logdir runs
# 浏览器打开 http://localhost:6006
```

**预期结果**:50 epoch 后验证准确率 91%~93%。若达不到 90%:① 检查增强是否加了 ② lr 试 0.1 ③ 层数加厚(每层 3 个 block)。

## 5. 学会看曲线(面试常被问:怎么判断训练状态)

| 现象 | 判断 | 处理 |
|---|---|---|
| loss 不降 | 欠拟合 / lr 太小 / 数据没打乱 | 加大 lr、检查 shuffle |
| loss 震荡发散 / NaN | lr 太大 | 降 lr、加梯度裁剪、检查数据有无异常值 |
| train 高 val 也高 | 欠拟合 | 更大模型、更多 epoch、调 lr |
| train 低 val 高 | **过拟合** | 增强、weight decay、dropout、早停 |
| train 低 val 低 | 理想状态 | 可以再训一会观察 |

## 6. 显存与速度小技巧(了解即可,面试加分)

- `pin_memory=True`:数据先锁页再拷 GPU,加速传输(已在上方代码里)
- AMP 混合精度:前向/激活用 bf16 省显存,权重梯度用 fp32 保精度(已在上方代码里)
- `torch.compile(model)`:图编译加速 10%~50%,一行代码,训练慢可以试
- 梯度累积:显存不够、想模拟大 batch 时用(教程 03 的 GPT 训练里有完整例子)
- `DataLoader(persistent_workers=True)`:workers 不反复重启,省时间

## 7. 本阶段常见坑总结

- [ ] `model(x)` 前忘记 `.to(device)`,报 device mismatch
- [ ] `model.eval()` 没调,验证时 BN/Dropout 行为错误
- [ ] `zero_grad` 忘调,梯度累加导致不收敛
- [ ] 验证循环没包 `inference_mode`,显存浪费
- [ ] num_workers>0 但没写 `if __name__ == "__main__"`,Windows 报错
- [ ] checkpoint 只存了 `model.state_dict()`,断点续训时 optimizer 状态丢失

## 完成标志

- [ ] Tensor 速查和 autograd 四问都理解
- [ ] train_cifar.py 亲手抄过并跑通(不是复制了事)
- [ ] CIFAR-10 达到 90%+,tensorboard 曲线截图保存
- [ ] 仓库 + README(含曲线截图)push 到 GitHub
- [ ] 能口头解释:训练循环每一行在干什么、AMP 原理、为什么验证要 eval 模式
