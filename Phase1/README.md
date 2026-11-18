# dataset-stats

一个统计 JSONL 数据集基本信息的命令行小工具(学习 Python 工程化的练手项目)。

## 安装
pip install -r requirements.txt

## 用法
python dataset_stats.py data.jsonl --max-show 5

## 示例
INFO:__main__:读取完成,共 1 条
{
  "样本数": 1,
  "字段": [
    {
      "名称": "instruction",
      "覆盖率": "100.0%",
      "唯一值数": 1,
      "最常见": [
        [
          "你好",
          1
        ]
      ]
    },
    {
      "名称": "output",
      "覆盖率": "100.0%",
      "唯一值数": 1,
      "最常见": [
        [
          "你好!有什么可以帮你?",
          1
        ]
      ]
    }
  ]
}

## 测试
pytest