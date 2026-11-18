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
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)

def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL 文件，返回 Python 对象。"""
    with open(path, encoding="utf-8") as f:
        for line_no, line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)                
            except json.JSONDecodeError as e:
                logger.error(f"第 {line_no + 1} 行 JSON 解析失败: {e}")


def describe(rows: Iterable[Dict[str, Any]], max_show: int = 5) -> Dict[str, Any]:
    """统计 JSONL 数据集的样本数与字段分布。

    Args:
        rows: JSONL 数据集的迭代器。
        max_show: 每个字段显示的最大值数量。
    """