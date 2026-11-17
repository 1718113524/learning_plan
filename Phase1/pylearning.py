# def read_lines(path):
#     with open(path, encoding='utf-8') as f:
#         for line in f:
#             yield line.strip()

# def count_words(lines):
#     for line in lines:
#         words = line.split()
#         yield len(words)

# for n in count_words(read_lines('example.txt')):
#     print(n)
# import os
# print(os.name)

# import time
# from functools import wraps

# def timer(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         print(result)
#         end_time = time.perf_counter()
#         print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
#         return result
#     return wrapper

# @timer
# def slow_function():
#     time.sleep(0.2)
#     return "Finished"

# slow_function()

# from dataclasses import dataclass, field

# @dataclass
# class Trainconfig:
#     learning_rate: float
#     batch_size: int=32
#     num_epochs: int=10
#     tags: list[str]=field(default_factory=list)

# cfg = Trainconfig(learning_rate=0.001, tags=['experiment1', 'baseline'])
# print(cfg)

# import json

# data = {"name": "Alice", "age": 30, "city": "New York", "tags" :["nlp", "llm", "ml"]}
# text = json.dumps(data, indent=4)
# print(text)

# # JSON 字符串 → Python 对象
# obj = json.loads(text)

# # 直接写入文件
# with open("phase1/config.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# # 直接从文件读取
# with open("phase1/config.json", encoding="utf-8") as f:
#     loaded = json.load(f)
#     print(loaded)


# import json

# # 写:一行一条
# rows = [{"instruction": "你好", "output": "你好!有什么可以帮你?"}]
# with open("phase1/data.jsonl", "w", encoding="utf-8") as f:
#     for row in rows:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")

# # 读:逐行惰性读取(几 GB 也不爆内存)
# def read_jsonl(path):
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 yield json.loads(line)

# for row in read_jsonl("phase1/data.jsonl"):
#     print(row["instruction"])


import argparse

parser = argparse.ArgumentParser(description="训练脚本")
parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--debug", action="store_true")   # 布尔开关

args = parser.parse_args()
print(args.lr, args.epochs, args.debug)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.debug("细节调试信息")    # 默认不显示
logger.info("正常流程信息")
logger.warning("可能有问题")
logger.error("出错了")
