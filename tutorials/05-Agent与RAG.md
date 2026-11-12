# 教程 05:Agent 与 RAG(W10-11)

> 对应计划 Phase 5。目标:手写一个最小 ReAct Agent + 一个 RAG 问答,不依赖框架。

---

## 1. LLM API 调用基础(OpenAI 兼容接口)

> 几乎所有国产模型(DeepSeek、Qwen、Kimi、GLM)都提供 OpenAI 兼容接口,只改 base_url 和模型名。这也是你以后调任何 API 的通用写法。

```bash
pip install openai
```

```python
# api_basics.py
from openai import OpenAI

client = OpenAI(
    api_key="sk-你的key",
    base_url="https://api.deepseek.com/v1",   # 以 DeepSeek 为例,换成任意厂商
)

# 1. 普通对话
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
)
print(resp.choices[0].message.content)

# 2. 流式输出(打字机效果 + 首 token 更快)
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一首五言绝句"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")

# 3. 异常重试(生产环境必备)
import time
for attempt in range(3):
    try:
        resp = client.chat.completions.create(model="deepseek-chat",
                                              messages=[{"role": "user", "content": "hi"}])
        break
    except Exception as e:
        print(f"第 {attempt + 1} 次失败: {e}")
        time.sleep(2 ** attempt)     # 指数退避:2s, 4s, 8s
```

## 2. Function Calling(结构化输出的正道)

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "北京今天天气怎么样?"}]
resp = client.chat.completions.create(
    model="deepseek-chat", messages=messages, tools=tools,
)
msg = resp.choices[0].message
if msg.tool_calls:                      # 模型决定调用工具
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)   # {"city": "北京"}
    # 执行真实函数,把结果以 role="tool" 回传
    messages.append(msg)                # assistant 消息(带 tool_calls)
    messages.append({"role": "tool", "tool_call_id": call.id,
                     "content": "北京今天晴,25°C"})
    resp2 = client.chat.completions.create(model="deepseek-chat", messages=messages)
    print(resp2.choices[0].message.content)
```

要点:
- function calling = 模型输出结构化 JSON + 你执行 + 结果回传,**本质是让 LLM 填参数表**
- JSON mode(`response_format={"type": "json_object"}`)是另一条路:让输出是合法 JSON,自己解析,适合简单结构化抽取

## 3. RAG:本地知识库问答(最小可跑)

```bash
pip install chromadb
```

```python
# rag.py —— 切分 → 向量化 → 检索 → 生成
import chromadb
from openai import OpenAI

client = OpenAI(api_key="sk-你的key", base_url="https://api.deepseek.com/v1")

docs = [
    "哈尔滨是黑龙江省省会,以冰雪文化和俄式建筑闻名,每年冬天举办冰雪大世界。",
    "北京大学软件与微电子学院以工程实践为导向培养软件人才。",
    # ... 放你自己的文档
]

def get_embedding(text: str) -> list[float]:
    """用 embedding 接口把文本转成向量。也可本地跑 bge-m3 等开源模型。"""
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

# 切分:固定长度 + 重叠(简单版;生产可换更好的切分器)
chunks = []
for i, doc in enumerate(docs):
    for j in range(0, len(doc), 400):          # 步长 400,窗口 500 → 重叠 100 字
        chunks.append((f"{i}-{j}", doc[j:j + 500]))

db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_or_create_collection("docs")
col.add(ids=[c[0] for c in chunks],
        embeddings=[get_embedding(c[1]) for c in chunks],
        documents=[c[1] for c in chunks])

def ask(question: str, top_k: int = 3) -> str:
    hits = col.query(query_embeddings=[get_embedding(question)], n_results=top_k)
    context = "\n---\n".join(hits["documents"][0])
    prompt = (f"根据以下资料回答问题,资料里没有的信息直接说不知道。\n"
              f"资料:\n{context}\n\n问题:{question}")
    r = client.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

print(ask("哈尔滨有什么特色?"))
```

### RAG 面试必答题

- **完整链路**:文档 → 切分(chunking)→ embedding → 向量库 → 检索(top-k)→ 生成(带上下文提示词)
- **切分策略**:固定长度+重叠 / 按段落标题 / 语义切分;chunk 太大丢精度、太小丢上下文,500-1000 token 是常用起点
- **常见坑**:检索不准(embedding 不匹配领域、chunk 不合适)→ 加 reranker(如 bge-reranker)重排;幻觉(资料里没有硬编)→ 提示词约束 + 引文标注;知识时效 → 更新向量库
- **RAG vs 微调**:RAG 注入"知识"(外部、可更新、可溯源),微调改变"行为/格式"(内部能力);知识密集型选 RAG,风格对齐选微调

## 4. ✅ M5 项目:手写最小 ReAct Agent

> 不依赖任何框架,完整走通「思考 → 调用工具 → 观察 → 再思考」循环。跑通后你就能在面试里说:"Agent 的核心循环我手写过,框架只是工程化包装。"

```python
# react_agent.py —— 手写最小 ReAct Agent(约 80 行)
import re

from openai import OpenAI

client = OpenAI(api_key="sk-你的key", base_url="https://api.deepseek.com/v1")

SYSTEM = """你是一个会使用工具的助手。可用工具:
- calculate(表达式): 计算数学表达式
- search(关键词): 搜索信息

需要工具时,严格按格式输出:
Thought: 我的思考
Action: calculate(1+1)
(不要输出其他内容)

拿到 Observation 后继续思考。能够回答时输出:
Thought: 我的思考
Final Answer: 最终答案"""


def calculate(expr: str) -> str:
    """只允许数字和运算符,拒绝注入。"""
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "非法表达式"
    return str(eval(expr))


def search(query: str) -> str:
    # 占位实现:接 DuckDuckGo / SerpAPI / 公司内部检索即可
    return f"关于「{query}」的搜索结果:(占位,请替换为真实搜索 API)"


TOOLS = {"calculate": calculate, "search": search}


def run_agent(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0)
        reply = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"--- 模型输出 ---\n{reply}")

        action = re.search(r"Action:\s*(\w+)\((.*?)\)", reply, re.DOTALL)
        if action:
            name, args = action.group(1), action.group(2)
            result = TOOLS[name](args)
            messages.append({"role": "user", "content": f"Observation: {result}"})
            print(f"Observation: {result}")
            continue                      # 拿到观察结果,继续下一轮思考
        if "Final Answer:" in reply:
            return reply.split("Final Answer:")[-1].strip()
    return "达到最大步数,任务失败"


if __name__ == "__main__":
    print(run_agent("(3+5)*2 等于多少?然后告诉我这个数的平方"))
```

**运行结果应该是**:模型先 `Action: calculate((3+5)*2)` → Observation: 16 → 再 `Action: calculate(16**2)` → Final Answer: 256。跑通后把 `search` 换成真实搜索 API,加一个"查天气"工具,就是你的 agent-demo 项目。

### Agent 面试必答题

- **ReAct 是什么**:Reasoning + Acting,交替"思考(选工具/推理)"与"行动(调用工具拿结果)",用观察结果驱动下一步,直到能回答
- **Agent 三件套**:规划(planning:任务拆解/反思)、记忆(memory:短时=对话上下文,长时=向量库)、工具(tools:function calling)
- **为什么不直接用 LangChain**:框架帮你管消息拼接、工具注册、多轮状态,但核心循环就 80 行;能说清"我手写核心、理解框架只是封装"是加分项
- **MCP 是什么**:Model Context Protocol,统一 LLM 应用连接外部工具/数据的协议(server 暴露工具,client 调用)。热点,面试前跑一个官方 demo(modelcontextprotocol.io)

## 5. LangChain / LlamaIndex 定位(了解,不深学)

| 框架 | 一句话定位 |
|---|---|
| LangChain | Agent 编排全家桶(链、工具、记忆、多智能体),生态最大 |
| LlamaIndex | 数据侧框架,强在索引/检索/RAG 工程 |

> 建议:本阶段**手写为主**,最后花半天把 demo 用 LangChain 重写一遍对比,面试时两个都能讲。

## 6. 提示工程速查(面试加分)

- **few-shot**:给 2-3 个示例再问,模型照猫画虎
- **CoT**:加一句"一步一步思考",数学/推理题准确率显著提升
- **结构化输出**:要 JSON 就用 JSON mode / function calling,不要靠祈使句
- **system prompt**:定角色、定边界(能干什么不能干什么)、定输出格式

## ✅ M5 完成清单

- [ ] API 调用 + 流式 + 重试跑通
- [ ] function calling 跑通,能解释 tool_calls 消息结构
- [ ] RAG demo 跑通,能讲完整链路与三个常见坑
- [ ] ReAct Agent 手写跑通,再加 1-2 个工具
- [ ] MCP 官方 demo 跑一遍
- [ ] 仓库 agent-demo(RAG + ReAct)push 到 GitHub,附 demo 截图
