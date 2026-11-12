# 教程 07:Kaggle 指南(W8 起并行)

> 对应计划 Phase 7。目的:① 检验学习成果拿硬证据 ② 免费 GPU 跑微调 ③ 简历上的实战经历。

---

## 1. 平台基础(30 分钟上手)

1. 注册 https://www.kaggle.com(建议用 Google 账号)
2. **Notebook**:云端 Jupyter,自带常见库(pytorch、transformers 都有)
3. **GPU 配额**:免费账号每周约 30 小时 GPU(通常 T4×2 或 P100),用完等下周刷新
4. **提交流程**:比赛页 → Submit Prediction → 上传预测文件 → 排行榜出分(Public LB)
5. **比赛结构**:每个比赛有 train.csv(带标签)/ test.csv(不带标签)/ sample_submission.csv(提交格式)
6. 关键概念:
   - **Public / Private LB**:最终排名看 Private(你没见过的数据),防止过拟合排行榜
   - **公开 notebook**:别人的完整方案,先抄基线再改进是标准玩法

## 2. 入门练手:Titanic(1-2 天,走通闭环)

```python
# titanic.py —— 最简基线,目标是"走通提交流程"而不是高分
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def preprocess(df):
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())        # 缺失值填充
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    for col in ["Sex", "Embarked"]:                          # 类别编码
        df[col] = LabelEncoder().fit_transform(df[col])
    return df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = preprocess(train)[features]
y = train["Survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pred = model.predict(preprocess(test)[features])
sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
sub.to_csv("submission.csv", index=False)
print("提交 submission.csv 到比赛页即可出分")
```

> 从「跑通提交」到「上分」的标准动作:EDA(pandas 看分布)→ 特征工程(家庭人数、称呼等)→ 交叉验证(5 折,不要只看 LB)→ 模型切换(LGBM/XGB)→ ensemble(平均/投票)。B 站/知乎搜"Titanic 特征工程"看 1-2 篇就够了。

## 3. LLM/NLP 比赛怎么打(以官网在办为准)

**常见题型与思路:**

| 题型(往届典型) | 核心思路 |
|---|---|
| 偏好预测(LMSYS Chatbot Arena) | 用 LLM 提取特征(长度、语法、关键词)+ 文本分类模型;LLM 直接打分做特征;ensemble |
| 提示词还原(LLM Prompt Recovery) | 生成式:微调小模型或 prompt 大模型生成被隐去的提示词 |
| 数学推理(AIMO) | 微调开源小模型 + 数据增强 + 多模型投票 |
| 对话策略(LLM 20 Questions) | Agent 策略:关键词猜测、决策树、用 RL 训练提问策略 |
| 效率挑战(LLM Efficiency Challenge) | 微调小模型逼近大模型效果,考核显存/速度,和你的 nanoGPT/微调技能直接对口 |

**标准流程(任何比赛通用):**
1. 找公开 notebook 里 upvote 最高的,复制跑通基线(拿到第一个有效分数)
2. 建立 5 折交叉验证(Trust CV,别信 LB)
3. 每次改动用表格记录:改动 / CV / LB / 备注
4. 后期:ensemble 多个差异大的模型
5. 赛后写总结:方案 + 关键上分点 + 失败尝试 → 发博客/放简历

**心态**:目标是「完整参赛 + 有 CV 意识 + 会看 notebook」,不是奖牌。奖牌规则按官网,铜牌通常要求进入一定排名比例且比赛有一定规模,不强求。

## 4. 用 Kaggle 免费 GPU 跑微调(一举两得)

1. Create → New Notebook → 右侧 Settings:
   - Accelerator 选 **GPU T4 x2**
   - Internet 打开
2. 在 notebook 里直接:
   ```python
   !pip install -q transformers datasets peft trl accelerate
   # 然后粘贴教程 04 的 train_sft.py 代码,把 data_files 换成你的数据路径
   ```
3. 数据上传:比赛页/数据集页上传 zip → notebook 里 `!unzip` 解压;微调数据几 MB 直接挂数据集
4. 训练完保存 adapter:
   ```python
   model.save_pretrained("/kaggle/working/qwen0.5b-lora")
   # 打包下载:右侧 Output 面板 → 选中文件夹 → Download
   ```
5. 注意:免费 GPU 单次会话上限约 9 小时,中途断线会丢 `/kaggle/working`(非 working 目录的已提交版本才持久)。**训练代码和 checkpoint 定期存成版本(Save Version),或及时下载。**

> Kaggle 的 T4×2(共 32GB)跑 QLoRA 微调 0.5B-1.5B 完全够用;Colab 免费版只有单 T4 且经常断,长训练优先 Kaggle。

## 5. 赛后总结模板(直接可发博客)

```markdown
# [比赛名]赛后总结
- 赛题与评估指标:
- 我的最终方案:(数据 / 特征 / 模型 / ensemble 一句话概括)
- 关键上分点:1. ... 2. ...
- 失败的尝试:1. ...(为什么没用)
- 最终排名:Public xx / Private xx(共 xxx 队)
- 收获:...
```

## 6. 时间盒(别让比赛吃掉主线)

- 每周固定 3-4 小时给 Kaggle,平时少碰
- 比赛截止前两周才允许加时间
- 主线(教程 03/04 的项目)永远优先——**项目是简历主体,Kaggle 是锦上添花**

## 完成标志

- [ ] 注册 + Titanic 提交出分
- [ ] 选定一场 LLM/NLP 赛,复现公开基线
- [ ] 建立 CV,用表格记录每次实验
- [ ] 完整打完 1-2 场,写赛后总结
- [ ] 用 Kaggle GPU 跑通过一次微调
