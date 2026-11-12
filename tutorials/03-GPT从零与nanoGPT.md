# 教程 03:从零重写 GPT + 吃透 nanoGPT(W5-7)

> 对应计划 Phase 3。目标:解决你「nanoGPT 读不懂、知识融会贯通不了」的问题。
> 方法:先手敲一个最小 GPT(本文第 1-4 节),跑通训练;再带着问题精读 nanoGPT(第 5-6 节)。

**建议学习顺序**
1. 看一遍 Karpathy「Let's build GPT: from scratch, in code, spelled out」视频(B 站搜"Karpathy GPT 从零"有搬运),建立整体画面
2. 手敲本文 gpt.py → data.py → train.py,在 TinyShakespeare 上训练到能生成莎士比亚风格的文本
3. 再打开 nanoGPT 源码逐行精读,你会发现「原来我都见过,只是它更工程化」

---

## 1. 完整 GPT 模型(gpt.py)

**先看架构全景,再逐行敲:**

```
输入 token ids (B, T)
  → Token Embedding + Position Embedding
  → ×N 个 Block[ LayerNorm → CausalSelfAttention → 残差 → LayerNorm → MLP → 残差 ]
  → 最终 LayerNorm → lm_head 线性层 → logits (B, T, vocab_size)
  → 与 targets 算 CrossEntropy(语言模型:预测下一个 token)
```

```python
# gpt.py —— 一个最小但完整的 GPT-2(结构对齐 nanoGPT,建议逐行手敲)
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 256      # 上下文长度
    vocab_size: int = 65       # 词表大小(TinyShakespeare 字符级是 65)
    n_layer: int = 6           # Transformer block 数
    n_head: int = 6            # 注意力头数
    n_embd: int = 384          # embedding 维度
    dropout: float = 0.2


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # 一次矩阵乘算 QKV
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 因果 mask:下三角矩阵。注册为 buffer = 不参与梯度,但随模型保存/迁移设备
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 因果 mask:把"未来"位置置 -inf,softmax 后概率为 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)   # 升维
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False) # 降维
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))    # pre-norm + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd, bias=False),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying:输入 embedding 与输出头共享权重(省参数 + 效果更好)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # 关键初始化:残差路径的输出层按 1/sqrt(2N) 缩放,抵消 N 层残差堆叠的方差增长
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)          # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # 语言模型目标:每个位置预测下一个 token,摊平成 (B*T, vocab) 算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]          # 超出上下文就截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)   # 温度缩放
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")      # top-k 过滤
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # 按概率采样
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        # weight decay 只作用于 2D 权重(Linear/Embedding),bias 和 LayerNorm 不衰减(见第 6 节)
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
```

**自检**:关掉参考,能从 0 重写出 `CausalSelfAttention.forward` 的 10 行和 `Block.forward` 的 4 行,才算真会。

## 2. 数据准备(data.py,字符级,最简单路径)

> 真实 GPT-2 用 BPE/tiktoken(词表 50257);字符级只有 65 个字符,是为了让你**先把整个训练流程跑通**。词表大小只是 config 的一个数字,流程完全一样。

```python
# data.py —— 下载 TinyShakespeare 并转成字符级 token
import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(data_dir: str = "data") -> dict:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    ckpt = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos,
            "train": data[:n], "val": data[n:]}
    torch.save(ckpt, os.path.join(data_dir, "tiny_shakespeare.pt"))
    print(f"词表大小 {vocab_size},训练 token 数 {n:,}")
    return ckpt


if __name__ == "__main__":
    prepare()
```

## 3. 训练脚本(train.py,含全部训练 trick)

```python
# train.py —— 训练最小 GPT:混合精度 + 梯度累积 + warmup/cosine + 梯度裁剪
import math
import time

import torch

from data import prepare
from gpt import GPT, GPTConfig

ckpt = prepare() if not __import__("os").path.exists("data/tiny_shakespeare.pt") else None
ckpt = torch.load("data/tiny_shakespeare.pt", weights_only=False)
train_data, val_data = ckpt["train"], ckpt["val"]
stoi, itos = ckpt["stoi"], ckpt["itos"]

config = GPTConfig(vocab_size=ckpt["vocab_size"])
model = GPT(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M,device = {device}")

batch_size, block_size = 32, config.block_size
grad_accum_steps, max_iters = 8, 5000
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # 目标 = 每个位置的下一个 token
    return x.to(device), y.to(device)


optimizer = model.configure_optimizers(weight_decay=1e-1, lr=3e-4)


def get_lr(it: int) -> float:
    if it < 500:                              # warmup:前 500 步线性升温
        return 3e-4 * (it + 1) / 500
    if it > 4500:                             # 尾部保持最低学习率
        return 3e-4 * 0.1
    progress = (it - 500) / (4500 - 500)
    return 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))   # 中间 cosine 衰减


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(100)
        for k in range(100):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


t0 = time.time()
for it in range(max_iters):
    for g in optimizer.param_groups:
        g["lr"] = get_lr(it)
    if it % 100 == 0:
        print(f"iter {it}: {estimate_loss()}")
    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum_steps):                      # 梯度累积模拟大 batch
        x, y = get_batch("train")
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps   # 关键:累积时除以步数,梯度才等价于大 batch
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # 裁剪爆炸梯度
    scaler.step(optimizer)
    scaler.update()

print(f"训练耗时 {(time.time() - t0) / 60:.1f} 分钟")

# 生成莎士比亚风格文本
model.eval()
start = "ROMEO:\n"
ids = torch.tensor([stoi[c] for c in start], dtype=torch.long, device=device).unsqueeze(0)
out_ids = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)[0].tolist()
print("".join(itos[i] for i in out_ids))
```

> 运行 `python train.py`。**CPU 也能跑**(6 层 384 维约 10M 参数,慢一些但 1-2 小时能看到 loss 下降);有 GPU 更快。观察:iter 0 时 val loss ≈ 4.1(= ln(65),词表均匀分布),随着训练降到 1.5 以下生成质量明显变好。

## 4. 采样三兄弟(必会讲)

- **temperature**:对 logits 除以温度 T 再 softmax。T→0 退化成 greedy(确定但重复);T 越大越随机(多样但胡言乱语)。T=1 是原始分布。
- **top-k**:只保留概率最高的 k 个 token,其余置 -inf 后重新采样。k 太小死板,k 太大失去意义。
- **top-p(核采样)**:按概率从大到小累加,保留累计概率刚超过 p 的最小集合,集合大小随分布自适应。
- 常用组合:`temperature=0.8, top_k=40`(GPT-2 官方默认),或 `temperature=0.7, top_p=0.9`。

## 5. nanoGPT 精读地图(带着上面 4 个文件的问题去读)

**model.py**
- `register_buffer("bias", ...)`:和你写的因果 mask 一样,但名字刻意叫 "bias" —— 为了兼容 FlashAttention 的 `is_causal` 判断逻辑
- `flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')` → `y = F.scaled_dot_product_attention(q, k, v, dropout_p=..., is_causal=True)`:PyTorch 2.0+ 内置融合注意力,等价于你手写的 mask+softmax,但快 2-3 倍且省显存
- residual 初始化 `std=0.02/math.sqrt(2 * config.n_layer)`:和你写的一样
- weight tying 后 `lm_head` 只在前向里用 `self.transformer.wte.weight` 直接转置乘,省一次线性层开销
- `crop_block_size`:为 KV cache 推理预留的上下文截断
- `estimate_mfu`:估算算力利用率(H100 上约 50% 是正常水平)

**train.py**
- `gradient_accumulation_steps * batch_size * block_size` = 一次优化步消耗的 token 数,这是"batch size"的真实定义
- 学习率调度:`warmup_iters` 线性升温 → `lr_decay_iters` 内 cosine 衰减 → 之后保持 `min_lr`(GPT-2 原配置 warmup 375M token,总 300B token,min_lr 10%)
- `if ddp:` 分支:DDP 多卡训练,`model = DDP(model, device_ids=[ddp_local_rank])`,数据用 DistributedSampler 切分
- `torch.compile`:`model = torch.compile(model)` 一行,图优化加速
- 训练循环核心 4 行(和你写的 train.py 完全对应):
  ```python
  loss = loss / gradient_accumulation_steps
  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
  ```
- 评估时 `ctx = nullcontext() if 'cuda' in device else torch.amp.autocast(...)`,且用 `model.eval()` 前后切换
- checkpoint 保存 `raw_model.state_dict()`(去掉 DDP 包装),并保留 optimizer 状态和配置,支持断点续训

**sample.py**
- 输入 prompt 编码 → `x = x[:, -model.config.block_size:]` 截断 → 循环生成
- `ctx = nullcontext() if 'cuda' in device else torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)`:CPU 也走 bf16 加速
- `if temperature < 1e-6: do_sample = False`(温度趋 0 就是 argmax);否则 `logits = logits / temperature`
- top-k 过滤逻辑和你写的一模一样

**prepare.py**
- 支持 openwebtext(真实 GPT-2 预训练数据,几十 GB)、TinyStories、字符级 shakespeare
- 用 tiktoken 的 GPT-2 BPE:词表 50257,每个 token 平均约 0.7 个英文单词——这就是 GPT-2 参数量 124M 的来源(50257×768 + 1024×768 + 12×12×768² ≈ 124M)

**参数量公式(面试常考)**:embedding(vocab×d)+ 位置(block×d)+ 每层(attn 4d² + mlp 8d²)= 12d²×层数,对 GPT-2-small:d=768,N=12 → 12×768²×12 ≈ 85M + embedding 39M ≈ 124M。

## 6. 训练细节十问(面试官最爱问,必须能答)

1. **为什么 weight decay 只作用于 2D 权重?** bias 和 LayerNorm 的 scale 是"加性/缩放"参数,decay 它们会伤害模型;2D 权重有缩放等价性(scale 权重等效于改初始化),decay 有明确的几何意义。
2. **为什么需要 warmup?** 训练初期梯度方差大,Adam 的二阶矩估计不准,大学习率会打乱刚学到的模式;线性升温让优化器先"摸底"。Transformer 对初期大 lr 尤其敏感。
3. **为什么 cosine 衰减?** 前期大步快速下降,后期小步收敛到平坦区域,平坦解泛化更好;比 step decay 平滑。
4. **为什么要梯度裁剪(1.0)?** 大模型训练中偶发 loss spike,单个大梯度会炸掉权重;clip 保证优化路径稳定。nanoGPT 默认 clip=1.0。
5. **为什么 loss 要除以累积步数?** 反向传播是线性的,平均梯度等价于真大 batch 的梯度;不除的话等效学习率被放大 accum 倍。
6. **bf16 和 fp16 的区别?** fp16 指数位少,数值范围小(最大 65504),需要 loss scaler 防止梯度下溢;bf16 和 fp32 指数位相同、精度低,一般不需要 scaler(代码里保留 scaler 是为了兼容)。
7. **HellaSwag 是什么?** 常识推理完形填空选择题,GPT-2 时代的标准评估集。nanoGPT 里 `eval_hellaswag.py` 算每句的似然选答案,和训练循环共用 forward。
8. **为什么位置编码用可学习的?** GPT 用的就是可学习绝对位置编码(最大 1024);现代模型常用 RoPE(旋转位置编码,天然支持相对位置和长度外推)——这个问题你要能讲出 RoPE 的存在和动机。
9. **flash attention 是什么?** IO-aware 的注意力实现:把 QKV 分块在 SRAM 里算,避免把大注意力矩阵写回 HBM。数学结果不变,显存从 O(n²) 降到 O(n),速度 2-3 倍。
10. **DDP 原理?** 每个 GPU 一份完整模型、不同数据;反向传播后 AllReduce 平均梯度再各自更新,所以每张卡的"等效 batch"要乘卡数。

## 7. ✅ M3 产出清单

- [ ] gpt.py / data.py / train.py 亲手敲完并跑通
- [ ] loss 曲线记录(iter 0 / 500 / 2000 / 5000 的 train/val loss)
- [ ] 生成样例保存(不同 temperature/top-k 各一段,对比效果)
- [ ] 用 tiktoken 版本再做一遍数据管线(`pip install tiktoken`,把字符级换成 GPT-2 BPE,其余不变)
- [ ] nanoGPT 四文件精读笔记(每行能说出为什么)
- [ ] 技术笔记《从零训练一个小 GPT 的完整过程》发布到博客/GitHub
- [ ] 仓库 gpt-from-scratch push 到 GitHub(自己重写的代码)
- [ ] 能口头回答第 6 节全部 10 问

> 想挑战:把 GPTConfig 改成 GPT-2 配置(block_size=1024, vocab=50257, n_layer=12, n_embd=768, n_head=12),用 Kaggle GPU 训练 124M 模型 —— 这就是 nanoGPT 的 speedrun。
