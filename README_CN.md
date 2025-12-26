# 双代理 PCG 系统

**基于大语言模型代理的零样本 3D 地图生成：用于过程内容生成 (PCG) 的双代理架构**

基于 [arXiv:2512.10501](https://arxiv.org/abs/2512.10501)

🌐 **[项目主页](https://had0yun.github.io/zero-shot-3d-map-generation-with-llm-agents/)** | 📄 [arXiv 论文](https://arxiv.org/abs/2512.10501)

[English](README.md) | [한국어](README_KR.md) | **中文**

## 概述 (Overview)

这是用于零样本过程内容生成 (PCG) 参数配置的双代理 Actor-Critic (参与者-评论者) 架构的完整实现。该系统使现成的大语言模型 (LLM) 无需进行针对特定任务的微调即可与 PCG 工具进行交互。

## 架构 (Architecture)

```
                     用户提示词 (P_user)
                            |
                            v
+------------------------------------------------------------------+
|                        编排器 (ORCHESTRATOR)                      |
|  +------------------------------------------------------------+  |
|  |                   上下文管理器 (CONTEXT MANAGER)            |  |
|  |  - 状态替换策略 (State-Replacement Strategy，固定大小缓冲区) |  |
|  |  - API 文档注入                                            |  |
|  |  - 使用示例管理                                            |  |
|  +------------------------------------------------------------+  |
|                            |                                      |
|      +---------------------+---------------------+               |
|      |                                           |               |
|      v                                           v               |
|  +-------------------+                   +-------------------+   |
|  |   Actor 代理      |      S_i          |   Critic 代理     |   |
|  |   (语义解释器)    | ----------------> |   (静态验证器)    |   |
|  |                   |                   |                   |   |
|  |                   | <---------------- |                   |   |
|  |  温度 (Temp):     |     反馈 (FB)     |  温度 (Temp):     |   |
|  |     0.4           |                   |     0.2           |   |
|  +-------------------+                   +-------------------+   |
|                            |                                      |
|                   +--------+--------+                            |
|                   |   批准通过？    |                            |
|                   +--------+--------+                            |
|                     是 (Yes) |   否 (No)                         |
|                      |       |     |                             |
|                      v       |     v                             |
|                  [返回]      | [迭代 或 尽力而为]                |
+------------------------------------------------------------------+
                            |
                            v
                   优化结果 (S_final)
```

## 安装 (Installation)

```bash
# 安装依赖项
pip install -r requirements.txt

# 设置你的 API 密钥
export ANTHROPIC_API_KEY=your-key-here  # Linux/Mac
set ANTHROPIC_API_KEY=your-key-here     # Windows
```

## 使用方法 (Usage)

### 使用真实 API 运行

```bash
python -m dual_agent_pcg.main
```

### 在模拟模式下运行（无需 API）

```bash
python -m dual_agent_pcg.main --mock
```

---

## 🚀 快速入门 - 直接复制此提示词！

**不想进行繁琐的设置？** 只需复制下面的提示词并将其粘贴到你最喜欢的 LLM 命令行界面（如 Claude、ChatGPT 等）中。将 `$ARGUMENTS` 替换为你的地图描述，即可开始使用！

<details>
<summary><strong>📋 点击以展开完整提示词</strong></summary>

```markdown
# 双代理 PCG 地图生成 (Dual-Agent PCG Map Generation)

你正在执行 **零样本双代理 PCG 优化协议 (Zero-shot Dual-Agent PCG Refinement Protocol)** (arXiv:2512.10501)。

## 用户请求 (P_user)

**地图描述**: $ARGUMENTS

---

## 协议 (Protocol)

你将在两个角色之间交替切换，直到收敛（最多 3 次迭代）：

### ACTOR 角色 (语义解释器)
生成 JSON 格式的参数轨迹序列 (Parameter Trajectory Sequence)。你必须：
- 将用户的意图转化为具体的 PCG 工具配置
- 包含具体的参数值（禁止使用如 "TBD" 之类的占位符）
- 所有的工具名称和参数必须基于下方的 API 文档
- 识别风险和假设

### CRITIC 角色 (静态验证器)  
根据文档评审轨迹。应用五维评审框架：
1. **工具选择 (Tool Selection)**：每个工具的名称是否完全匹配？
2. **参数正确性 (Parameter Correctness)**：所有必填参数是否齐全？数值是否在有效范围内？
3. **逻辑与顺序 (Logic & Sequence)**：生成器 (Generators) 是否在修改器 (Modifiers) 之前调用？依赖关系是否满足？
4. **目标对齐 (Goal Alignment)**：轨迹是否实现了用户的要求？
5. **完整性 (Completeness)**：是否有遗漏的步骤？

**保守策略 (CONSERVATIVE POLICY)**：仅在确定存在问题时才指出。

---

## 执行流程 (Execution Flow)

1. [ACTOR] 生成初始轨迹 S₀
2. [CRITIC] 评审 S₀ → 产生反馈
3. 如果发现问题且迭代次数 < 3:
   [ACTOR] 根据反馈修订轨迹 → S₁
   [CRITIC] 评审 S₁
   ... 重复直到批准通过或达到最大迭代次数
4. 输出最终批准的轨迹

---

## API 文档 (D)

### 生成器 (Generators)（必须在修改器之前调用）

#### CellularAutomataGenerator
创建有机陆地模式。适用于岛屿、大陆、洞穴。

**必填参数：**
- `width` (int): 网格宽度 [16, 256]
- `height` (int): 网格高度 [16, 256]  
- `fill_probability` (float): 初始填充率 [0.0, 1.0]
  - 0.3-0.4: 分散的陆地
  - 0.45-0.55: 均衡、连通的陆地
  - 0.6-0.7: 较大的、实心的陆地块
- `iterations` (int): 平滑处理次数 [1, 10]
- `birth_limit` (int): 出生阈值 [0, 8]（通常为 4）
- `death_limit` (int): 死亡阈值 [0, 8]（通常为 3）

**可选参数：** `seed` (int)

#### PerlinNoiseGenerator
创建平滑的高度图。适用于海拔、山脉、丘陵。

**必填参数：**
- `width` (int): 网格宽度 [16, 512]
- `height` (int): 网格高度 [16, 512]
- `scale` (float): 噪声缩放 [0.01, 1.0]
  - 0.01-0.03: 巨大且平滑的特征
  - 0.04-0.08: 适用于山脉
  - 0.1+: 粗糙且细节丰富
- `octaves` (int): 细节层数 [1, 8]
- `persistence` (float): 振幅衰减 [0.0, 1.0]

**可选参数：** `seed` (int), `lacunarity` (float, 默认 2.0)

### 修改器 (Modifiers)（在生成器之后应用）

#### HeightLayerModifier
创建离散的海拔区域。

**必填参数：**
- `layer_count` (int): 图层数量 [1, 10]
- `layer_heights` (list[float]): 阈值列表，升序排列
- `blend_factor` (float): 过渡平滑度 [0.0, 0.5]

#### ScatterModifier
在地形上散布物体。

**必填参数：**
- `object_type` (str): 以下之一："rock" (岩石), "tree" (树木), "grass_clump" (草丛), "bush" (灌木), "flower" (花卉)
- `density` (float): 散布密度 [0.0, 1.0]
- `valid_layers` (list[int]): 允许散布的图层索引列表

#### GrassDetailModifier
为特定图层添加草地覆盖。

**必填参数：**
- `target_layer` (int): 图层索引（从 0 开始）
- `coverage` (float): 覆盖百分比 [0.0, 1.0]

---

## 输出格式 (Output Format)

以 JSON 格式输出最终批准的轨迹：

{
  "final_trajectory": {
    "trajectory_summary": "<概述>",
    "tool_plan": [
      {
        "step": 1,
        "objective": "<此步骤实现的目标>",
        "tool_name": "<精确的工具名称>",
        "arguments": { ... },
        "expected_result": "<成功标准>"
      }
    ],
    "risks": ["<潜在问题>"]
  }
}

---

## 开始协议 (BEGIN PROTOCOL)

现在为以下内容执行双代理优化：**$ARGUMENTS**

首先由 [ACTOR] 生成初始轨迹 S₀。
```

</details>

> 💡 关于带有使用示例的完整提示词，请参阅 [`.opencode/command/Map.md`](.opencode/command/Map.md)

---

## OpenCode 集成（推荐）

**无需 API 密钥！** 直接在 OpenCode CLI 中使用双代理 PCG 系统。

### 快速设置

1. 将 `.opencode` 目录复制到你的用户主目录：

```bash
# Windows
xcopy /E /I .opencode %USERPROFILE%\.opencode

# Linux/Mac
cp -r .opencode ~/.opencode
```

2. 运行 OpenCode 并使用 `/Map` 命令：

```bash
opencode -c
```

```
/Map volcanic island with central crater lake
```

### 工作原理

OpenCode 集成利用已认证的 Claude 会话，消除了对单独 API 密钥的需求：

```
用户: /Map mountain terrain with 3 elevation zones
         │
         ▼
    ┌─────────────────┐
    │  Map.md 命令    │ ← 编排协议执行
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ Actor   │     │ Critic  │
│ 代理    │────▶│ 代理    │
│ (t=0.4) │◀────│ (t=0.2) │
└─────────┘     └─────────┘
             │
             ▼
    JSON 参数输出
```

### OpenCode 文件结构

```
.opencode/
├── agent/
│   ├── pcg-actor.yaml    # Actor 代理 (温度: 0.4)
│   └── pcg-critic.yaml   # Critic 代理 (温度: 0.2)
└── command/
    └── Map.md            # 带有 API 文档的主斜杠命令
```

### 对比：Python API vs OpenCode

| 特性 | Python API | OpenCode |
|---------|-----------|----------|
| 需要 API 密钥 | ✅ 是 | ❌ **否** |
| 设置复杂度 | pip install + 环境变量 | 复制文件夹 |
| 温度控制 | ✅ 0.4/0.2 | ✅ 0.4/0.2 |
| Token 追踪 | 精确 | 估算 |
| 使用方式 | 脚本/代码 | `/Map` 命令 |

---

### 🔄 自动化视觉反馈循环（Unity MCP 集成）

当 Unity MCP 可用时，系统现在支持**完全自动化的优化**。参数生成、地图执行、截图捕获和视觉比较的整个循环自动运行：

```
┌─────────────────────────────────────────────────────────────────────┐
│               自动化视觉反馈循环（4个阶段）                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  阶段1：参数生成（Dual-Agent）                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  /Map [参考图片.png]                                         │   │
│  │  → Actor Agent (t=0.4): 生成初始参数                         │   │
│  │  → Critic Agent (t=0.2): 验证并优化                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  阶段2：自动执行（Unity MCP）                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  → 通过 MCP 将参数应用到 TileWorldCreator                     │   │
│  │  → 自动执行地图生成                                           │   │
│  │  → 无需手动干预                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  阶段3：视觉比较（Dual-Agent）                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  → 通过 Unity MCP 捕获截图                                    │   │
│  │  → Comparison Actor: 分析原图 vs 生成结果                     │   │
│  │  → Comparison Critic: 验证相似度评估                          │   │
│  │  → 输出: 相似度分数 (0-100%)                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  阶段4：自动优化决策                                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  若相似度 ≥ 80%: ✅ 完成 - 输出最终参数                        │   │
│  │  若相似度 < 80%: 🔄 自动优化并重复（最多3次）                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 自动化循环要求

| 要求 | 说明 |
|------|------|
| **Unity MCP** | 必须安装并配置（[Unity MCP 设置](https://github.com/anthropics/unity-mcp)） |
| **TileWorldCreator** | 用于程序化地形生成的 Unity 插件 |
| **Unity 项目** | 需打开并加载 TileWorldCreator 场景 |

#### 工作原理

1. **您提供**: 参考图片或文字描述
2. **系统处理**: 参数生成 → 执行 → 截图 → 比较 → 优化
3. **您获得**: 最终优化的参数（相似度 ≥ 80% 或3次迭代后的最佳结果）

#### 回退：手动模式

如果 Unity MCP 不可用，系统回退到**手动模式**：
- 仅生成参数
- 用户手动应用到 PCG 工具
- 用户提供结果截图进行优化

#### 使用示例

```bash
# 完全自动化（使用 Unity MCP）
/Map ~/reference/mountain_village.png
# → 系统自动生成、执行、比较和优化
# → 最终输出: 优化的 JSON 参数

# 手动回退（无 Unity MCP）
/Map ~/reference/mountain_village.png
# → 系统生成参数
# → 用户手动应用，提供截图
/Map ~/screenshots/attempt1.png "优化: 山的高度需要更大"
```

#### 相似度阈值

| 分数 | 动作 |
|------|------|
| **≥ 80%** | ✅ 接受 - 参数被认为是最优的 |
| **60-79%** | 🔄 自动优化 - 调整参数并重新生成 |
| **< 60%** | 🔄 大幅优化 - 显著的参数变更 |

系统在返回最佳结果之前自动迭代**最多3次**。

---

### 程序化用法 (Programmatic Usage)

```python
import asyncio
from dual_agent_pcg.models import SystemConfig
from dual_agent_pcg.llm_providers import create_provider
from dual_agent_pcg.orchestrator import Orchestrator

async def main():
    # 与论文匹配的配置（第 4.1 节）
    config = SystemConfig(
        actor_temperature=0.4,   # 平衡创意与指令遵循
        critic_temperature=0.2,  # 提供一致且自信的反馈
        max_iterations=1         # 论文中的设置
    )
    
    # 创建提供者
    llm = create_provider("anthropic", api_key="your-key")
    
    # 创建编排器
    orchestrator = Orchestrator(
        config=config,
        llm_provider=llm,
        api_documentation="...",  # 你的 PCG 工具文档
        usage_examples=["..."]    # 轨迹示例
    )
    
    # 执行
    result = await orchestrator.execute("Create a mountain terrain...")
    
    if result.success:
        print("批准的轨迹:", result.final_trajectory)
    else:
        print("尽力而为的结果:", result.final_trajectory)

asyncio.run(main())
```

## 项目结构 (Project Structure)

```
dual_agent_pcg/
├── .opencode/                    # OpenCode 集成（新增！）
│   ├── agent/
│   │   ├── pcg-actor.yaml       # Actor 代理配置 (t=0.4)
│   │   └── pcg-critic.yaml      # Critic 代理配置 (t=0.2)
│   └── command/
│       └── Map.md               # /Map 斜杠命令
├── __init__.py                   # 包初始化
├── models.py                     # Pydantic 模型 (ActorOutput, CriticFeedback 等)
├── prompts.py                    # Actor 和 Critic 的系统提示词
├── llm_providers.py              # LLM 提供者抽象（Anthropic, OpenAI, Mock）
├── orchestrator.py               # 算法 1 的实现
├── main.py                       # 可运行示例
├── config.yaml                   # 配置文件
└── requirements.txt              # 依赖项
```

## 论文配置 (Paper Configuration)（第 4.1 节）

| 设置 | 数值 | 依据 |
|---------|-------|-----------|
| LLM 模型 | Claude 4.5 Sonnet | 论文的选择 |
| Actor 温度 | 0.4 | "平衡创意与指令遵循" |
| Critic 温度 | 0.2 | "提供一致且自信的反馈" |
| 最大迭代次数 | 1 | "所有试验均设为 1" |

## 核心组件 (Key Components)

### Actor 代理（语义解释器）
- 将自然语言转换为参数轨迹序列 (Parameter Trajectory Sequence)
- 输出：`{trajectory_summary, tool_plan, risks}`
- 禁止直接执行工具

### Critic 代理（静态验证器）
- 根据文档评估轨迹
- 五维评审框架：
  1. 工具选择 (Tool Selection)
  2. 参数正确性 (Parameter Correctness)
  3. 逻辑与顺序 (Logic & Sequence)
  4. 目标对齐 (Goal Alignment)
  5. 完整性 (Completeness)
- 保守策略：仅在确定存在阻塞性问题时才标记

### 上下文管理器 (Context Manager)
- 实现状态替换策略 (State-Replacement Strategy)（第 3.3 节）
- 固定大小的上下文缓冲区
- 覆盖之前的轨迹（而非追加）

## 算法 1：零样本双代理 PCG 优化 (Zero-shot Dual-Agent PCG Refinement)

```
输入: P_user, D, E, K
输出: S_final

Context_actor <- {P_user, D, E}
S_0 <- Actor(Context_actor)
i <- 0

while i < K do:
    Feedback <- Critic(S_i, D, E)
    if Feedback = empty then
        return S_i
    Context_actor <- UpdateContext(S_i, Feedback)
    S_{i+1} <- Actor(Context_actor)
    i <- i + 1

return S_K
```

## 许可证 (License)

本实现仅用于教育和研究目的。

## 引用 (Citation)

```bibtex
@article{her2025zeroshot3dmap,
  title={Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation},
  author={Her, Lim Chien and Yan, Ming and Bai, Yunshu and Li, Ruihao and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.10501},
  year={2025}
}
```
