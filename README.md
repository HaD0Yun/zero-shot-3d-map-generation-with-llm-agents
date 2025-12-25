# Dual-Agent PCG System

**Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation**

Based on [arXiv:2512.10501](https://arxiv.org/abs/2512.10501)

**English** | [한국어](./README_KR.md)

## Overview

This is a complete implementation of the Dual-Agent Actor-Critic architecture for zero-shot Procedural Content Generation (PCG) parameter configuration. The system enables off-the-shelf LLMs to interface with PCG tools without task-specific fine-tuning.

## Architecture

```
                     User Prompt (P_user)
                            |
                            v
+------------------------------------------------------------------+
|                        ORCHESTRATOR                               |
|  +------------------------------------------------------------+  |
|  |                   CONTEXT MANAGER                          |  |
|  |  - State-Replacement Strategy (fixed-size buffer)          |  |
|  |  - API Documentation injection                             |  |
|  |  - Usage Examples management                               |  |
|  +------------------------------------------------------------+  |
|                            |                                      |
|      +---------------------+---------------------+               |
|      |                                           |               |
|      v                                           v               |
|  +-------------------+                   +-------------------+   |
|  |   ACTOR AGENT    |      S_i          |   CRITIC AGENT   |   |
|  |   (Semantic      | ----------------> |   (Static        |   |
|  |    Interpreter)  |                   |    Verifier)     |   |
|  |                  | <---------------- |                   |   |
|  |  Temperature:    |     Feedback      |  Temperature:    |   |
|  |     0.4          |                   |     0.2          |   |
|  +-------------------+                   +-------------------+   |
|                            |                                      |
|                   +--------+--------+                            |
|                   |  Approve?       |                            |
|                   +--------+--------+                            |
|                     Yes    |    No                               |
|                      |     |     |                               |
|                      v     |     v                               |
|                  [RETURN]  | [ITERATE or BEST EFFORT]            |
+------------------------------------------------------------------+
                            |
                            v
                   RefinementResult (S_final)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your-key-here  # Linux/Mac
set ANTHROPIC_API_KEY=your-key-here     # Windows
```

## Usage

### Run with Real API

```bash
python -m dual_agent_pcg.main
```

### Run in Mock Mode (No API Required)

```bash
python -m dual_agent_pcg.main --mock
```

---

## OpenCode Integration (Recommended)

**No API key required!** Use the Dual-Agent PCG system directly within OpenCode CLI.

### Quick Setup

1. Copy the `.opencode` directory to your user home:

```bash
# Windows
xcopy /E /I .opencode %USERPROFILE%\.opencode

# Linux/Mac
cp -r .opencode ~/.opencode
```

2. Run OpenCode and use the `/Map` command:

```bash
opencode -c
```

```
/Map volcanic island with central crater lake
```

### How It Works

The OpenCode integration leverages the authenticated Claude session, eliminating the need for a separate API key:

```
User: /Map mountain terrain with 3 elevation zones
         │
         ▼
    ┌─────────────────┐
    │  Map.md Command │ ← Orchestrates the protocol
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐     ┌─────────┐
│  Actor  │     │ Critic  │
│  Agent  │────▶│  Agent  │
│ (t=0.4) │◀────│ (t=0.2) │
└─────────┘     └─────────┘
             │
             ▼
    JSON Parameter Output
```

### OpenCode Files Structure

```
.opencode/
├── agent/
│   ├── pcg-actor.yaml    # Actor agent (temperature: 0.4)
│   └── pcg-critic.yaml   # Critic agent (temperature: 0.2)
└── command/
    └── Map.md            # Main slash command with API docs
```

### Comparison: Python API vs OpenCode

| Feature | Python API | OpenCode |
|---------|-----------|----------|
| API Key Required | ✅ Yes | ❌ **No** |
| Setup Complexity | pip install + env vars | Copy folder |
| Temperature Control | ✅ 0.4/0.2 | ✅ 0.4/0.2 |
| Token Tracking | Exact | Estimated |
| Usage | Script/Code | `/Map` command |

---

### Programmatic Usage

```python
import asyncio
from dual_agent_pcg.models import SystemConfig
from dual_agent_pcg.llm_providers import create_provider
from dual_agent_pcg.orchestrator import Orchestrator

async def main():
    # Configuration matching paper (Section 4.1)
    config = SystemConfig(
        actor_temperature=0.4,   # Balance creativity and adherence
        critic_temperature=0.2,  # Consistent, confident feedback
        max_iterations=1         # Paper setting
    )
    
    # Create provider
    llm = create_provider("anthropic", api_key="your-key")
    
    # Create orchestrator
    orchestrator = Orchestrator(
        config=config,
        llm_provider=llm,
        api_documentation="...",  # Your PCG tool docs
        usage_examples=["..."]    # Example trajectories
    )
    
    # Execute
    result = await orchestrator.execute("Create a mountain terrain...")
    
    if result.success:
        print("Approved trajectory:", result.final_trajectory)
    else:
        print("Best effort:", result.final_trajectory)

asyncio.run(main())
```

## Project Structure

```
dual_agent_pcg/
├── .opencode/                    # OpenCode integration (NEW!)
│   ├── agent/
│   │   ├── pcg-actor.yaml       # Actor agent config (t=0.4)
│   │   └── pcg-critic.yaml      # Critic agent config (t=0.2)
│   └── command/
│       └── Map.md               # /Map slash command
├── __init__.py                   # Package initialization
├── models.py                     # Pydantic models (ActorOutput, CriticFeedback, etc.)
├── prompts.py                    # System prompts for Actor and Critic
├── llm_providers.py              # LLM provider abstraction (Anthropic, OpenAI, Mock)
├── orchestrator.py               # Algorithm 1 implementation
├── main.py                       # Runnable example
├── config.yaml                   # Configuration file
└── requirements.txt              # Dependencies
```

## Paper Configuration (Section 4.1)

| Setting | Value | Rationale |
|---------|-------|-----------|
| LLM Model | Claude 4.5 Sonnet | Paper's choice |
| Actor Temperature | 0.4 | "Balance creativity with instruction adherence" |
| Critic Temperature | 0.2 | "Consistent and confident feedback" |
| Max Iterations | 1 | "Set to one for all trials" |

## Key Components

### Actor Agent (Semantic Interpreter)
- Translates natural language to Parameter Trajectory Sequence
- Output: `{trajectory_summary, tool_plan, risks}`
- Forbidden from executing tools directly

### Critic Agent (Static Verifier)
- Evaluates trajectories against documentation
- 5-Dimension Review Framework:
  1. Tool Selection
  2. Parameter Correctness
  3. Logic & Sequence
  4. Goal Alignment
  5. Completeness
- Conservative policy: Only flags blocking issues when certain

### Context Manager
- Implements State-Replacement Strategy (Section 3.3)
- Fixed-size context buffer
- Overwrites previous trajectory (not append)

## Algorithm 1: Zero-shot Dual-Agent PCG Refinement

```
Input: P_user, D, E, K
Output: S_final

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

## License

This implementation is for educational and research purposes.

## Citation

```bibtex
@article{her2025zeroshot3dmap,
  title={Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation},
  author={Her, Lim Chien and Yan, Ming and Bai, Yunshu and Li, Ruihao and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.10501},
  year={2025}
}
```
