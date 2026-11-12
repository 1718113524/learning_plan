# 教程 04:LoRA 微调与对齐 —— 后训练方向核心(W8-9)⭐

> 对应计划 Phase 4,**整个计划优先级最高的项目**。目标:跑通 Qwen2.5-0.5B 的 LoRA-SFT 全流程,并理解 LoRA/DPO/RLHF 原理到能面试。

---

## 1. HuggingFace 30 分钟入门

```bash
pip install transformers datasets accelerate
```

```python
# quickstart.py —— 加载模型、对话模板、生成
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "用三句话解释什么是 LoRA"}]
# chat template:把对话结构转成模型训练/推理时的标准格式(必会)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("输入模型的文本:\n", text)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print("输出:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

要点:
- `from_pretrained` 会下载模型到本地缓存(0.5B 约 1GB,国内可用 `HF_ENDPOINT=https://hf-mirror.com` 加速)
- `apply_chat_template` 是 SFT 数据的核心:训练格式必须和推理格式一致
- 显存不够 CPU 也能跑 0.5B,只是慢

## 2. LoRA 原理(面试必考,先懂公式再写代码)

**公式**:冻结原权重 W,只训练低秩增量 ΔW = B·A(A: d×r,B: r×d,r ≪ d),前向为 `h = Wx + BAx`(或 `h = (W + BA)x`)。

**为什么低秩可行**(两个主流解释,都要能讲):
1. 预训练权重是"过参数化"的,下游任务适配发生在低秩子空间(Intrinsic Dimensionality 实验:几百维就能达到全参微调 90% 效果)
2. 增量矩阵的奇异值衰减很快,大头信息在少数奇异值上

**参数量怎么算**:每个被 LoRA 化的矩阵 d_in×d_out,LoRA 参数量 = r×(d_in + d_out)。
例:Qwen2.5-0.5B,d=896,只挂 q/k/v/o 四个 proj(各 896×896),r=8:
每矩阵 8×(896+896)=14336 → 每层 4×14336≈57k → 24 层 ≈ **1.4M**,占 0.5B 的约 0.28%。

**α 是什么**:实际增量是 (α/r)·BA。α/r 是缩放系数,α=r 时退化为标准 BA。常用 r=8,α=16(即 2 倍缩放)。

**target_modules 选什么**:q,k,v,o 四件套是保守默认;效果更好可以加 gate_proj/up_proj/down_proj(MLP 部分),参数量翻倍但效果更好。

## 3. SFT 数据(alpaca 风格 JSONL)

```jsonl
{"instruction": "用三句话介绍哈尔滨", "output": "哈尔滨是黑龙江省省会,被誉为冰城。它以冰雪文化和俄式建筑闻名。冬季的冰雪大世界吸引着全国游客。"}
{"instruction": "1+1=?", "output": "等于 2。"}
{"instruction": "把下面的句子翻译成英文:今天天气很好。", "output": "The weather is nice today."}
```

数据要点(面试会问"你怎么构造微调数据"):
- 每条要有明确的 instruction,output 是期望的完整回答
- 多样性 > 数量:几百条高质量、覆盖目标能力的数据比几万条重复数据有效
- 可以从 alpaca 中文数据集(开源)开始,再自己写 100-200 条你关心的场景

## 4. ✅ M4 项目:完整 LoRA-SFT 脚本

```bash
pip install peft trl
```

**train_sft.py**

```python
"""Qwen2.5-0.5B LoRA 指令微调(peft + transformers Trainer)。"""
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token     # Qwen 无 pad token,用 eos 代替

# 显存不足(如 T4 16G / 本地小显存)就把 bfloat16 换 float16,或走 QLoRA(第 6 节)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # 先四件套跑通
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 观察:可训参数只有 ~0.3%

# 数据 → chat template → tokenize
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

def format_example(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

dataset = dataset.map(lambda e: {"text": format_example(e)})

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

args = TrainingArguments(
    output_dir="./output/qwen0.5b-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # 等效 batch size 32
    num_train_epochs=3,
    learning_rate=2e-4,                 # LoRA 常用 1e-4 ~ 3e-4
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),     # 无 GPU 自动回退 fp32
    report_to="none",
)
trainer = Trainer(
    model=model, args=args, train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
model.save_pretrained("./output/qwen0.5b-lora-final")   # 只存 LoRA 增量,几 MB
```

**infer_compare.py(微调前后对比,面试/README 的素材)**

```python
"""同一批问题,对比 base 模型与 LoRA 模型的回答。"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
questions = [
    "用三句话介绍哈尔滨。",
    "帮我写一封请假邮件,100 字以内。",
    "解释一下什么是过拟合。",
]

def chat(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
lora = PeftModel.from_pretrained(base, "./output/qwen0.5b-lora-final")

for q in questions:
    print("=" * 60)
    print("问题:", q)
    print("--- 微调前 ---")
    print(chat(base, tokenizer, q))
    print("--- 微调后 ---")
    print(chat(lora, tokenizer, q))
```

> 把这些对比整理成表格放 README(问题 / 微调前 / 微调后)。**这就是简历上"微调提升了 xx"的证据来源。**

## 5. 怎么评估微调效果(面试必问)

1. **人工对比**:固定 30-50 个测试问题,微调前后输出并排看,统计"更好/持平/变差"比例
2. **能力保持**:微调不能毁掉原模型通用能力,抽几道常识/数学题确认没有灾难性遗忘
3. **简单 benchmark**:小型评测集(如本地跑几条 MMLU 样本、自己造 50 道领域问答)
4. **训练曲线**:train loss 平稳下降、最终不过低(过低=过拟合训练集,泛化差)

## 6. QLoRA(4bit 量化微调,小显存必备)

只需改两处:

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # 权重 4bit 加载
    bnb_4bit_quant_type="nf4",                # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,    # 计算时反量化到 bf16
    bnb_4bit_use_double_quant=True,           # 二次量化,进一步省显存
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config)
model = prepare_model_for_kbit_training(model)   # 加梯度检查点等适配
model = get_peft_model(model, lora_config)
# TrainingArguments 里加: optim="paged_adamw_8bit"
```

原理速记:**NF4**(对正态分布的权重更优的 4bit 分桶)+ **双重量化**(量化常数也量化)+ **分页优化器**(优化器状态换页防 OOM)。0.5B 模型 4bit 权重约 0.3GB,LoRA 训练显存 < 4GB,免费 T4 随便跑。

## 7. 对齐:RLHF 与 DPO

### RLHF 三阶段(要能画图讲)

```
① SFT:指令数据监督微调 → 学会"对话格式 + 遵循指令"
② RM:训练奖励模型,给"回答质量"打分(用人类偏好排序数据)
③ PPO:用 RM 当奖励函数,强化学习优化策略(在线采样、需要 reference model 防止跑偏)
```

### DPO(公式要能写出来)

```
L_DPO = -E[ log σ( β·log(πθ(y_w|x)/πref(y_w|x)) - β·log(πθ(y_l|x)/πref(y_l|x)) ) ]
```

- y_w 偏好回答,y_l 被拒回答,σ 是 sigmoid,β 控制偏离参考模型的强度
- **直觉**:让"偏好回答相对参考模型的优势"大于"被拒回答的优势",直接用偏好对做分类式优化
- **DPO vs PPO**:DPO 不需要训练 RM、不需要在线采样,稳定、便宜、快,是目前工业界最常用的对齐算法;PPO 理论上在线采样覆盖更广、效果上限更高,但训练复杂不稳定。面试标准答案:**"DPO 把 RLHF 的 RM+RL 两步化简成一步偏好分类,便宜稳定;但离线数据覆盖有限,大厂重要模型仍会用 RL 类方法。"**

### DPO 实操(trl 库,最小可跑)

```bash
pip install trl
```

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# dpo_data.jsonl 每行三列: prompt / chosen / rejected
dataset = load_dataset("json", data_files="dpo_data.jsonl", split="train")

# model / ref_model / tokenizer 准备与 SFT 相同(模型用 SFT 后的 adapter 更标准)
trainer = DPOTrainer(
    model=model, ref_model=ref_model,        # ref_model 冻结不动
    args=DPOConfig(output_dir="./output/dpo", beta=0.1,
                   per_device_train_batch_size=2, learning_rate=5e-5,
                   bf16=torch.cuda.is_available(), report_to="none"),
    train_dataset=dataset, tokenizer=tokenizer,
)
trainer.train()
```

> 数据从哪来:网上有开源偏好对数据集(如 hh-rlhf 子集、中文偏好数据),或自己写几十条"同一个问题,一个好答案一个差答案"——DPO 数据格式就是 prompt/chosen/rejected 三列。

## 8. 推理与部署概念速查(面试常问,理解概念即可)

| 概念 | 一句话 |
|---|---|
| KV cache | 推理时缓存历史 token 的 K/V,避免每步重算;显存随长度线性增长:2×层数×头数×头维×长度×字节数 |
| GPTQ | 训练后量化,逐层用二阶信息(海森矩阵)找最优量化权重,4bit 效果好 |
| AWQ | 激活感知量化:识别重要权重通道,放大保护后再量化,比 GPTQ 快 |
| GGUF | llama.cpp 生态的量化格式(如 Q4_K_M),CPU 推理友好,Ollama 用它 |
| vLLM | 高吞吐推理引擎,核心是 PagedAttention(分页管理 KV cache) |
| safetensors | 比 pickle 安全(无代码执行)且加载快,模型权重标准格式 |

```bash
# 体验一次 vLLM 部署(有 GPU 时):
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct
# 然后就可以像调 OpenAI API 一样调它(见教程 05)
```

## 9. 显存估算速算(面试手算题)

| 内容 | 每个参数占用 |
|---|---|
| 权重 | fp32=4B,fp16/bf16=2B,int8=1B,int4=0.5B |
| 梯度 | 与权重同精度 |
| Adam 状态(m+v) | 8B(fp32 存) |

- 全参微调 0.5B bf16:权重 1GB + 梯度 1GB + Adam 4GB ≈ **6GB**(还要加激活,T4 16G 刚好)
- LoRA 微调 0.5B:权重 1GB + LoRA 梯度/状态 ≈ 0.1GB ≈ **1.2GB 左右**
- QLoRA:权重 0.3GB + 增量 ≈ **< 1GB**
- 训练 7B 全参:14 + 14 + 56 ≈ 84GB → 必须 4×A100;QLoRA 7B ≈ 6-8GB → 单卡可跑

## ✅ M4 完成清单

- [ ] quickstart 跑通,能解释 chat template
- [ ] LoRA 公式、参数量计算、α/r 缩放能写出来
- [ ] SFT 全流程跑通(建议数据量 500-2000 条)
- [ ] 微调前后对比表整理进 README
- [ ] QLoRA 版本跑通,记录显存变化
- [ ] DPO 公式推导笔记 + trl 最小实验跑通
- [ ] 仓库 qwen-lora-sft push 到 GitHub
- [ ] 能口头讲:RLHF 三阶段、DPO vs PPO、QLoRA 三个技巧、KV cache 显存公式
