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