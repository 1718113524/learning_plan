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