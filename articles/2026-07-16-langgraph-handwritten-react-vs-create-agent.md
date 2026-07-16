# LangGraph 手写 ReAct vs create_agent：一次工程权衡

## 一行能搞定的事，为什么我写了两百行

我在滴滴做后端，日常维护一个日均百万级调用的模型推理服务。业余时间写了个开源项目 [agent-demo](https://github.com/masonhe132/agent-demo)，里面有一对刻意做成"镜像"的文件：

- `agent.py`：用 LangChain 的 `create_agent`，核心代码不到十行
- `graph_agent.py`：用 LangGraph 的 `StateGraph` 手写同样的 ReAct Agent，两百多行

两个文件对外接口完全一致，跑同一组测试用例，输出行为一样。看起来后者纯属没事找事——但这个项目后来长出了 `hitl_agent.py`（人工确认中断）、`multi_agent.py`（Supervisor 多智能体）、`app.py`（WebSocket 流式服务），全部长在手写版的骨架上，没有一个能从 `create_agent` 那条路自然生长出来。

这篇文章把这次权衡完整摊开：什么时候一行封装够用，什么时候必须下沉到图编排层，以及为什么我建议每个做 Agent 的工程师都手写一遍再回头用封装。

先交代一下动机。我的本职工作是维护线上推理链路，对"框架黑箱"这件事有职业性警惕：白天排查一个 P99 毛刺，如果中间某一跳的逻辑藏在框架里看不见，定位时间就会从分钟级涨到小时级。所以当我开始做 Agent 应用时，第一反应不是"哪个框架最快出效果"，而是"哪条路线在出问题时我能自救"。这个视角贯穿了整个项目的技术选型。

## ReAct 循环到底在循环什么

先把概念钉死。ReAct（Reason + Act）不是什么玄学，就是一个带退出条件的 while 循环：

```
Think：LLM 看到当前所有消息，决定"我能直接回答吗，还是需要调工具"
Act：  如果需要，发出 tool_calls，执行工具
Observe：工具结果作为新消息塞回上下文
回到 Think，直到 LLM 不再发 tool_calls
```

所有 Agent 框架，无论包装得多花哨，拆到底都是这个循环。区别只在于：循环的控制权在谁手里。`create_agent` 把循环藏在框架内部；`StateGraph` 让你亲手把循环画成一张图。

这里有个容易被忽略的点：ReAct 和传统 RAG 的本质区别不在检索，而在决策权。纯 RAG 是固定流水线——检索、拼 prompt、生成，每一步都是写死的；ReAct 把"要不要检索、要不要计算、要不要再来一轮"的决策权交给了 LLM。我的项目里 RAG 就是被降级成了 Agent 的一个普通工具 `search_docs`，LLM 觉得需要查文档才调它，问"今天几号"时根本不会碰向量库。这就是所谓 "RAG as a Tool"：比纯 RAG 灵活，比裸 Agent 准确。

理解了这一点，再看图编排就清楚了：我们要画的图，就是给这个决策循环搭一个显式的执行框架。

## 手写版走读：一张图的四个零件

`graph_agent.py` 的结构就是 LangGraph 的最小完备集：State、Node、Edge、compile。挨个看。

### 1. State：Agent 的记忆载体

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

四行代码，但 `Annotated[list, add_messages]` 是 LangGraph 最核心的机制之一——reducer。当某个节点返回 `{"messages": [新消息]}` 时，框架不是替换整个列表，而是追加到末尾。没有这个 reducer，每个节点都得自己拼接历史消息，状态管理立刻变成一摊泥。

更重要的是：**State 是可扩展的**。这里只有一个 `messages` 字段，但在同项目的 `multi_agent.py` 里，State 多了一个 `next_worker: str` 字段用于存放 Supervisor 的路由决策。用 `create_agent` 你碰不到 State 的定义，想加字段没有入口。

### 2. LLM 节点：只负责思考，不负责决策走向

```python
llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "qwen-plus"),
    temperature=0,
).bind_tools(tools)

def llm_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    return {"messages": [response]}
```

两个细节值得注意。第一，`bind_tools` 只是把工具的 schema 声明给 LLM，不执行任何东西——LLM 回复时在 `tool_calls` 字段里表达"我想调哪个工具"。第二，system prompt 是每次调用时现拼的，不进 State。State 里只存用户和 AI 的消息，prompt 属于运行时配置，这个分离在后面换 prompt 做 A/B 时会感谢自己。

节点函数刻意不判断"接下来去哪"。思考和路由分离，是图编排的基本纪律。

### 3. 工具节点：直接用官方轮子

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

手写不等于所有东西都手搓。`ToolNode` 干的事是纯体力活：读最后一条 AI 消息的 `tool_calls`，逐个执行，把结果包装成 `ToolMessage`。这里没有定制需求，用内置的。**手写路线的意义在于掌握组装权，而不是重造每个零件。**

### 4. 条件边：ReAct 循环的开关本体

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

整个 ReAct 循环的控制逻辑就这五行。LLM 的最新回复带 `tool_calls` 就去执行工具，不带就结束。所谓"Agent 自主决策"，工程上就是这一个 if。

把它写成独立函数还有个附带收益：**可以单测**。构造一条带 `tool_calls` 的假消息传进去，断言返回 `"tools"`，不用调一次真 LLM。项目里 23 个 pytest 用例 1 秒跑完、零 API 成本，就是靠这种"只测自己写的编排逻辑，不测 LLM 黑盒"的策略。

### 5. 组装与编译

```python
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue)
graph.add_edge("tools", "llm")

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)
```

注意 `graph.add_edge("tools", "llm")` 这条普通边：工具执行完必须回到 LLM，因为工具只执行不决策，结果够不够、要不要再调一次，得让 LLM 再看一眼。这条边加上前面的条件边，闭环就成了。

`compile(checkpointer=MemorySaver())` 开启对话记忆。调用时传 `thread_id`，同一个 thread_id 共享历史：

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config={"configurable": {"thread_id": thread_id}},
)
```

checkpointer 的机制值得多说两句，因为它是后面 HITL 和多用户隔离的地基。每执行完一个节点，LangGraph 就把当前 State 按 thread_id 存一份快照。`thread_id` 就是会话 ID：demo 里我给每个交互会话生成一个 UUID，Web 服务里则是每个 WebSocket 连接一个，天然实现多用户对话隔离。`MemorySaver` 是内存级实现，进程一退历史就没了，学习和演示够用；生产环境换成 SqliteSaver 或 PostgresSaver，一行改动，State 就落了盘——这意味着服务重启后对话能接着聊，甚至能回放某个 thread 的完整决策轨迹做审计。

跑一下验证记忆是否生效。demo 里的四连问：

```python
demo_questions = [
    "我叫小贺，请记住我的名字",
    "帮我算一下 256 * 3",
    "我刚才让你算的结果再乘以 2 是多少？",  # 需要记住上一轮的 768
    "我叫什么名字？",                        # 需要记住第一轮的信息
]
```

第三问要求 Agent 记住上一轮的计算结果 768 再乘 2，第四问要求跨三轮找回名字。手写版和封装版跑这组用例的输出完全一致——这也是我敢说两个实现"功能等价"的依据。

## 对照组：create_agent 的十行版本

`agent.py` 里等价的实现：

```python
from langchain.agents import create_agent

def build_agent():
    llm = ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "qwen-plus"),
        temperature=0,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
```

必须承认：它非常好。接口干净，checkpointer 照样支持，`invoke` 的调用方式和手写版一模一样。上面手写的 State、节点、条件边、循环边，`create_agent` 内部全帮你做了——它本质上就是官方替你组装好的那张图。

如果需求止步于"一个能调工具、有记忆的对话 Agent"，用它，别犹豫。问题出在需求不止步的时候。

而 Agent 类需求的特点恰恰是：几乎从不止步。我这个项目的演进路线很典型——第一周只想要个能算数、能查文档的对话机器人；第二周产品视角冒出来了："工具执行前能不能让人确认一下？"（HITL）；第三周："能不能让用户看到它在想什么，别转半天圈？"（流式）；第四周："能不能一个 Agent 管文档、一个管工具，互相配合？"（Multi-Agent）。每一个需求单看都不过分，但每一个都要求你能改图的拓扑。

## 手写路线的复利：三个延伸实现

这部分是权衡的关键证据，展开说说手写骨架上长出来的三个东西。

**条件路由 + HITL（`hitl_agent.py`）。** 这个版本把单一入口拆成了意图分流：`router_node` 先判断用户意图，`route_by_intent` 条件边把请求分发到 `chat_node`（闲聊直答）、`rag_node`（文档检索）、`tool_llm_node`（工具调用）三条路径。闲聊不必过工具判断，省一次 LLM 决策的延迟和成本。更关键的是工具链路上插了一个 `human_confirm_node`：调用 LangGraph 的 `interrupt()`，图在这里冻结，State 存进 checkpoint，函数外的世界拿到一条"等待确认"的提示。用户点了确认或拒绝之后，外部调用 `agent.invoke(Command(resume=True), config)` 或 `Command(resume=False)`，图从断点原地恢复。整个暂停-恢复过程跨进程调用都成立，因为状态在 checkpointer 里而不在内存栈上。

**流式输出（`stream_agent.py`）。** 用 `astream_events(version="v2")` 订阅图执行的细粒度事件流：`on_chat_model_stream` 是 LLM 每吐一个 token 推一次，`on_tool_start` / `on_tool_end` 标记工具执行的起止。`app.py` 里把这些事件翻译成自定义 JSON 协议经 WebSocket 推给前端，前端按事件类型决定是往当前气泡追加文字，还是新开一个"正在调用 calculator..."的工具气泡。用户能实时看到 Agent 的思考和动作，而不是面对一个转了八秒的 loading。

**Multi-Agent（`multi_agent.py`）。** Supervisor 模式：一个不绑定任何工具的 LLM 节点当调度员，看完整对话历史后输出一个强制 JSON 的路由决策 `{"next": "rag_worker"}`，条件边按 `next_worker` 字段分发给对应 Worker。两个设计点：Worker 干完活回边到 Supervisor 而不是直接结束，让调度员确认"问题都答完了吗"；JSON 解析失败或返回了不认识的 Worker 名时兜底到 FINISH，防止无限循环。State 里为此加了一个字段——前面说过，手写版的 State 是你自己定义的 TypedDict，加字段就是加一行。

这三个文件加起来的增量工作，都建立在 `graph_agent.py` 里那套 State/Node/Edge 的心智模型上。骨架没换过。

## 工程权衡对照表

| 维度 | create_agent | StateGraph 手写 |
|---|---|---|
| 上手成本 | 十行起跑，五分钟出 demo | 需要理解 State/Node/Edge/reducer，约 200 行 |
| 控制粒度 | 黑箱，循环逻辑不可见 | 每个节点、每条边显式可控 |
| 条件路由 / 意图分流 | 无入口 | 加 router 节点 + 条件边即可（`hitl_agent.py` 的 `route_by_intent` 把请求分流到 chat / rag / tool 三条路径） |
| Human-in-the-loop | 做不了 | 在工具节点前插 `human_confirm_node`，调 `interrupt()` 暂停图，外部用 `Command(resume=True/False)` 恢复 |
| 记忆 / 多用户隔离 | 支持（传 checkpointer） | 同样支持，且能按 thread_id 检查 `get_state` 快照、判断图是否被中断暂停 |
| 流式输出 | 能拿到 token 流 | `astream_events(version="v2")` 拿到细粒度事件：`on_chat_model_stream` / `on_tool_start` / `on_tool_end`，可翻译成自定义前端协议 |
| 调试可观测性 | 出问题只能看最终 messages | 每个节点是普通 Python 函数，可打断点、打日志、单测路由函数 |
| 复杂拓扑（Multi-Agent） | 不适用 | Supervisor 节点输出 JSON 路由决策，Worker 完成后回边到 Supervisor，循环 + 分支随便画 |
| 升级维护 | 跟着框架 API 走，高层 API 历史上变动不小 | 依赖的是 StateGraph 底层原语，更稳定；坏了也知道坏在哪个节点 |

表里最值钱的三行是 HITL、条件路由和调试。我在推理服务上踩过足够多的坑，深知生产系统的需求从来不是"跑起来"，而是"出问题时三分钟内定位到哪一步"。黑箱封装在 demo 阶段是效率，在排障阶段是负债。

HITL 尤其值得强调，因为它在真实业务里几乎是硬需求。Agent 要执行一个有副作用的操作——发赔偿、改库存、给用户发消息——没有哪个负责人敢让它全自动跑。这个能力的前提是你能"在任意两个节点之间插东西"，而 `create_agent` 的图是编译好交付的，没有缝。

再比如降本。路由判断这种"输出一个分类标签"的简单任务完全可以换小模型（qwen-turbo 级别），主推理流程才用 qwen-plus。手写版里 router 节点和 llm 节点是两个独立函数，各绑各的模型实例，改两行的事；封装版里只有一个 model 参数，做不到分级。在百万级日调用量的场景下，这种"简单任务用小模型"的分流能砍掉相当可观的一块账单——这是我在本职工作里反复验证过的规律，Agent 应用没有任何理由例外。

还有测试。项目里 23 个 pytest 用例分三层：工具层直接 `.invoke()` 测正常、边界、错误输入（calculator 里带了代码注入防御，这也要测到）；图结构层测 `should_continue`、`route_by_intent` 这类路由函数和 StateGraph 拓扑，不调真 LLM；Web 层用 `fastapi.testclient.TestClient` 测 HTTP 路由和 WebSocket 握手。1 秒跑完，零 LLM 调用成本。这套策略的前提同样是手写：路由逻辑是独立的纯函数才可测，藏在框架里就只能靠端到端调真模型碰运气。LLM 输出本身的不确定性怎么办？不测。只测自己写的编排逻辑，模型黑盒交给评估集去管，两件事别混在一个测试套件里。

## 我的结论

**Demo 快跑、内部工具、需求就是标准 ReAct：用 `create_agent`。** 别为了炫技手写，十行能解决的事写两百行是负资产。

**生产系统、需要定制路由 / 中断 / 复杂拓扑 / 精细排障：直接上 `StateGraph`。** 不要抱着"先用封装快速上线，以后有需求了再重构"的幻想——封装版和手写版的代码结构完全不同，那不是重构，是重写。我的项目里从 `graph_agent.py` 演进到 `hitl_agent.py` 只需要加节点加边，State 定义和 `chat(question, thread_id)` 的调用接口原样复用；如果起点是 `create_agent`，这条演进路径不存在，你会在第一个定制需求到来时推倒重来。

判断标准可以再收敛一点，就问三个问题：这个 Agent 会执行有副作用的操作吗？会有第二种意图路径吗？出了 bad case 需要复现决策过程吗？三个都是"否"，用封装；有一个"是"，手写。按我的经验，凡是要接进真实业务流程的 Agent，三问很难全否。

另外说一句公道话：这不是"封装 vs 手写"的站队问题，LangGraph 官方自己也是这么设计的——`create_agent` 底层就是一张标准 StateGraph，高层 API 服务快速验证，底层原语服务深度定制，两层各司其职。要警惕的不是封装本身，而是只会用封装。

**学习路径上，我强烈建议先手写一遍再用封装。** 顺序反了会有一种危险的错觉：你以为自己会 Agent 了，其实只是会调一个函数。手写一遍之后，`create_agent` 在你眼里不再是魔法，而是"一张我自己也画得出来的图"——这时候用封装才是真正的降本，因为你知道它替你省了什么，也知道它的边界在哪。用我在代码注释里的比喻：`create_agent` 是买整机，`StateGraph` 是自己装机。装过一次机的人买整机，和没装过的人买整机，是两种消费者。

完整代码（含 HITL、流式、Multi-Agent、FastAPI + WebSocket 服务和三层测试）在这里：

**https://github.com/masonhe132/agent-demo**

欢迎 issue 交流。
