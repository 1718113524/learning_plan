# 教程 01:Python 工程与 Git(W1-2)

> 对应计划 Phase 1。原则:**只补工程短板,深度学习理论跳过**。
> 每个知识点 15 分钟内看完代码、自己敲一遍;全部学完后完成 M1 项目。

---

## 1. 核心语法速成(挑你写简单代码时没用过的)

### 生成器(大文件处理的基石)

```python
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()   # yield 让函数变成生成器:惰性求值,不占内存

def count_words(lines):
    for line in lines:
        yield len(line.split()) # 生成器可以链式组合,像流水线

for n in count_words(read_lines("data.txt")):
    print(n)
```

要点:普通函数用 `return` 一次性返回;生成器用 `yield` 一个个吐出,处理几 GB 文件也不爆内存。

### 装饰器(面试高频:能写一个计时装饰器)

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)   # 保留原函数的名字和文档字符串
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timer
def slow_add(a, b):
    time.sleep(0.1)
    return a + b

slow_add(1, 2)   # 自动打印耗时
```

### dataclass(配置类、数据类的标配写法)

```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 10
    tags: list[str] = field(default_factory=list)   # 可变默认值必须用 default_factory

cfg = TrainConfig(epochs=20)
print(cfg)   # 自动生成 __repr__: TrainConfig(lr=0.0003, epochs=20, tags=[])
```

### 魔术方法(让自己的类支持 len / 索引 / 迭代)

```python
class Dataset:
    def __init__(self, items):
        self.items = items
    def __len__(self):            # 支持 len()
        return len(self.items)
    def __getitem__(self, i):     # 支持 dataset[i]、切片、for 循环
        return self.items[i]
    def __repr__(self):           # print 时显示友好信息
        return f"Dataset({len(self)} items)"
```

### 异常处理

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    print(f"出错:{e}")
else:
    print("没有异常才执行")
finally:
    print("无论如何都执行")

class ConfigError(Exception):     # 自定义异常
    pass
```

---

## 2. 数据格式(你点名的 JSON 重点)

### JSON

```python
import json

# 对象 → 字符串(ensure_ascii=False 保证中文不乱码)
data = {"name": "张三", "tags": ["nlp", "llm"]}
text = json.dumps(data, ensure_ascii=False, indent=2)
print(text)

# 字符串 → 对象
obj = json.loads(text)

# 文件读写
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
with open("config.json", encoding="utf-8") as f:
    loaded = json.load(f)
```

### JSONL(一行一个 JSON,大模型数据集的标准格式)

```python
import json

# 写:一行一条
rows = [{"instruction": "你好", "output": "你好!有什么可以帮你?"}]
with open("data.jsonl", "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# 读:逐行惰性读取(几 GB 也不爆内存)
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

for row in read_jsonl("data.jsonl"):
    print(row["instruction"])
```

### YAML(训练配置)与 CSV

```python
# pip install pyyaml
import yaml
config = yaml.safe_load("""
lr: 3e-4
epochs: 10
model:
  name: gpt2
  layers: 6
""")
print(config["model"]["layers"])   # 6
```

```python
import csv
with open("result.csv", "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig 防 Excel 乱码
    writer = csv.writer(f)
    writer.writerow(["模型", "准确率"])
    writer.writerow(["resnet", 0.92])
```

---

## 3. 工程化四件套:argparse / logging / typing / pytest

### argparse(命令行参数)

```python
import argparse
parser = argparse.ArgumentParser(description="训练脚本")
parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--debug", action="store_true")   # 布尔开关
args = parser.parse_args()
print(args.lr, args.epochs, args.debug)
# 用法: python train.py --lr 1e-4 --epochs 20 --debug
```

### logging(用日志替代 print)

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.debug("细节调试信息")    # 默认不显示
logger.info("正常流程信息")
logger.warning("可能有问题")
logger.error("出错了")
```

### typing(类型注解)

```python
def normalize(scores: list[float]) -> dict[str, float | None]:
    total = sum(scores) or 1
    return {f"item_{i}": s / total for i, s in enumerate(scores)}
# pip install pyright 后在终端运行 pyright 检查你的文件
```

### pytest(单元测试)

```python
# test_utils.py
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
```
```bash
pip install pytest
pytest          # 自动发现 test_*.py,跑通显示绿色通过
```

### ruff(代码风格,一条命令)

```bash
pip install ruff
ruff check .    # 检查问题
ruff format .   # 自动格式化
```

### VS Code 断点调试(5 分钟学会)

1. 打开一个 .py 文件,在行号左侧点击 → 出现红点(断点)
2. 左侧栏点「运行和调试」(Ctrl+Shift+D)→ 点击「运行和调试」按钮
3. 程序停在断点处 → 顶部工具栏:F10 单步跳过 / F11 单步进入 / F5 继续
4. 左侧「变量」面板可以实时看每个变量的值

> 从今天起,遇到 bug 先断点调试,不要无脑 print。

---

## 4. ✅ M1 项目:数据集统计工具(完整代码)

把下面 3 个文件放到 `D:\plan\m1-dataset-stats\`,跑通、测试通过、push 到 GitHub。

**dataset_stats.py**

```python
"""数据集统计工具:统计 JSONL 数据集的样本数与字段分布。

用法:
    python dataset_stats.py data.jsonl --max-show 5 --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """惰性逐行读取 JSONL,损坏行告警并跳过。"""
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("第 %d 行解析失败: %s", line_no, exc)


def describe(rows: list[dict[str, Any]], max_show: int) -> dict[str, Any]:
    """统计每个字段的覆盖率、唯一值数、最常见取值。"""
    total = len(rows)
    field_counter: Counter[str] = Counter()
    value_counter: dict[str, Counter[str]] = {}
    for row in rows:
        for key, value in row.items():
            field_counter[key] += 1
            value_counter.setdefault(key, Counter())[str(value)] += 1

    report: dict[str, Any] = {"样本数": total, "字段": []}
    for key, count in field_counter.most_common():
        values = value_counter[key]
        report["字段"].append({
            "名称": key,
            "覆盖率": f"{count / total:.1%}",
            "唯一值数": len(values),
            "最常见": values.most_common(min(max_show, 5)),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 JSONL 数据集基本信息")
    parser.add_argument("file", type=Path, help="JSONL 文件路径")
    parser.add_argument("--max-show", type=int, default=5, help="每个字段展示几种常见值")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    rows = list(read_jsonl(args.file))
    logger.info("读取完成,共 %d 条", len(rows))
    print(json.dumps(describe(rows, args.max_show), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

**test_dataset_stats.py**

```python
import json
from pathlib import Path

import pytest

from dataset_stats import describe, read_jsonl


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """构造一个临时 JSONL 测试数据。"""
    path = tmp_path / "sample.jsonl"
    lines = [
        {"instruction": "你好", "output": "世界"},
        {"instruction": "1+1=?", "output": "2"},
    ]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
                    encoding="utf-8")
    return path


def test_read_jsonl(sample_file: Path) -> None:
    rows = list(read_jsonl(sample_file))
    assert len(rows) == 2
    assert rows[0]["instruction"] == "你好"


def test_describe(sample_file: Path) -> None:
    rows = list(read_jsonl(sample_file))
    report = describe(rows, max_show=5)
    assert report["样本数"] == 2
    names = {field["名称"] for field in report["字段"]}
    assert names == {"instruction", "output"}
```

**README.md**(模板,以后每个项目都用这个结构)

```markdown
# dataset-stats

一个统计 JSONL 数据集基本信息的命令行小工具(学习 Python 工程化的练手项目)。

## 安装
pip install -r requirements.txt

## 用法
python dataset_stats.py data.jsonl --max-show 5

## 示例
(贴一段真实输出)

## 测试
pytest
```

> 自己跑通一次:造一个 100 行的测试 JSONL → 运行脚本 → 看输出 → 跑 pytest → 全绿。

---

## 5. Git 速查(边用边查)

### 高频命令

| 命令 | 作用 |
|---|---|
| `git status` | 看当前状态(最常用,随时敲) |
| `git add 文件` / `git add .` | 暂存修改 |
| `git commit -m "feat: 说明"` | 提交 |
| `git log --oneline` | 看历史 |
| `git diff` | 看未暂存的改动 |
| `git checkout -b 分支名` | 建分支并切换 |
| `git switch 分支名` | 切分支 |
| `git merge 分支名` | 合并到当前分支 |
| `git remote add origin 仓库地址` | 关联远程(只需一次) |
| `git push -u origin main` | 推送到远程 |
| `git pull` | 拉取远程更新 |
| `git reset --hard HEAD~1` | 撤销上一次提交(慎用) |
| `git reflog` | 找回"丢掉的"提交 |

### 提交信息规范(Conventional Commits)

```
feat: 新增统计输出 JSON 功能
fix: 修复空文件崩溃的问题
docs: 更新 README
chore: 调整 .gitignore
```

### .gitignore(Python 标准模板)

```
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
```

### 解决一次真实冲突(必做练习)

1. 在 GitHub 网页上直接改一下你的 README(加一行字)并 commit
2. 本地也改同一行不同内容并 commit
3. `git pull` → 出现 CONFLICT 标记
4. 打开文件,把 `<<<<<<<` / `=======` / `>>>>>>>` 标记删掉,保留你想要的最终内容
5. `git add` + `git commit` → 冲突解决完成

### 提一个真实 PR(必做)

1. 在 GitHub 上 fork 任意一个你感兴趣的项目(推荐 nanoGPT,正好后面要用)
2. 把 fork 后的仓库 clone 到本地:`git clone git@github.com:你的用户名/nanoGPT.git`
3. 新建分支改点东西(哪怕 README 里改一个标点)
4. push 到你的 fork → GitHub 网页点 "Compare & pull request"
5. 写清楚改了什么的说明 → 提交 PR

> PR 被合并当然好,没人合并也没关系——**走通流程本身就是目的**,面试时可以说"我给 xx 项目提过 PR"。

## 完成标志

- [ ] 上面每个代码片段都亲手敲过、跑过
- [ ] M1 项目:脚本 + 测试 + README 完成,`pytest` 全绿
- [ ] M1 项目 push 到 GitHub
- [ ] 亲手解决了一次 git 冲突
- [ ] 向任意开源项目提交了一个 PR 
