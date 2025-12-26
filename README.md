# Dual-Agent PCG System

**Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation**

Based on [arXiv:2512.10501](https://arxiv.org/abs/2512.10501)

🌐 **[Project Page](https://had0yun.github.io/zero-shot-3d-map-generation-with-llm-agents/)** | 📄 [arXiv Paper](https://arxiv.org/abs/2512.10501)

**English** | [한국어](./README_KR.md) | [中文](./README_CN.md)

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

## 🚀 Quick Start - Just Copy This Prompt!

**Too lazy to set everything up?** Just copy the prompt below and paste it into your favorite LLM CLI (Claude, ChatGPT, etc.). Replace `$ARGUMENTS` with your map description and you're good to go!

<details>
<summary><strong>📋 Click to expand the full prompt</strong></summary>

```markdown
# Dual-Agent PCG Map Generation

You are executing the **Zero-shot Dual-Agent PCG Refinement Protocol** (arXiv:2512.10501).

## User Request (P_user)

**Map Description**: $ARGUMENTS

---

## Protocol

You will alternate between two roles until convergence (max 3 iterations):

### ACTOR ROLE (Semantic Interpreter)
Generate a Parameter Trajectory Sequence as JSON. You must:
- Translate the user's intent into specific PCG tool configurations
- Include concrete parameter values (NO placeholders like "TBD")
- Ground all tool names and parameters in the API Documentation below
- Identify risks and assumptions

### CRITIC ROLE (Static Verifier)  
Review the trajectory against documentation. Apply the 5-dimension framework:
1. **Tool Selection**: Does each tool exist exactly as named?
2. **Parameter Correctness**: All required params present? Values in valid range?
3. **Logic & Sequence**: Generators before modifiers? Dependencies satisfied?
4. **Goal Alignment**: Does trajectory achieve user's requirements?
5. **Completeness**: Any missing steps?

**CONSERVATIVE POLICY**: Only flag issues you're CERTAIN about.

---

## Execution Flow

1. [ACTOR] Generate initial trajectory S₀
2. [CRITIC] Review S₀ → produce feedback
3. IF issues found AND iteration < 3:
   [ACTOR] Revise trajectory based on feedback → S₁
   [CRITIC] Review S₁
   ... repeat until approved or max iterations
4. Output final approved trajectory

---

## API Documentation (D)

### Generators (must be called before modifiers)

#### CellularAutomataGenerator
Creates organic landmass patterns. Ideal for islands, continents, caves.

**Required Parameters:**
- `width` (int): Grid width [16, 256]
- `height` (int): Grid height [16, 256]  
- `fill_probability` (float): Initial fill [0.0, 1.0]
  - 0.3-0.4: scattered landmasses
  - 0.45-0.55: balanced, connected
  - 0.6-0.7: larger, solid masses
- `iterations` (int): Smoothing passes [1, 10]
- `birth_limit` (int): Birth threshold [0, 8] (typically 4)
- `death_limit` (int): Death threshold [0, 8] (typically 3)

**Optional:** `seed` (int)

#### PerlinNoiseGenerator
Creates smooth heightmaps. Ideal for elevation, mountains, hills.

**Required Parameters:**
- `width` (int): Grid width [16, 512]
- `height` (int): Grid height [16, 512]
- `scale` (float): Noise scale [0.01, 1.0]
  - 0.01-0.03: large, smooth features
  - 0.04-0.08: good for mountains
  - 0.1+: rough, detailed
- `octaves` (int): Detail layers [1, 8]
- `persistence` (float): Amplitude falloff [0.0, 1.0]

**Optional:** `seed` (int), `lacunarity` (float, default 2.0)

### Modifiers (apply after generators)

#### HeightLayerModifier
Creates discrete elevation zones.

**Required Parameters:**
- `layer_count` (int): Number of layers [1, 10]
- `layer_heights` (list[float]): Thresholds, ascending order
- `blend_factor` (float): Transition smoothness [0.0, 0.5]

#### ScatterModifier
Scatters objects on terrain.

**Required Parameters:**
- `object_type` (str): One of "rock", "tree", "grass_clump", "bush", "flower"
- `density` (float): Scatter density [0.0, 1.0]
- `valid_layers` (list[int]): Layer indices to scatter on

#### GrassDetailModifier
Adds grass coverage to a layer.

**Required Parameters:**
- `target_layer` (int): Layer index (0-indexed)
- `coverage` (float): Coverage percentage [0.0, 1.0]

---

## Output Format

Output the FINAL APPROVED trajectory as JSON:

{
  "final_trajectory": {
    "trajectory_summary": "<overview>",
    "tool_plan": [
      {
        "step": 1,
        "objective": "<what this achieves>",
        "tool_name": "<EXACT tool name>",
        "arguments": { ... },
        "expected_result": "<success criteria>"
      }
    ],
    "risks": ["<potential issues>"]
  }
}

---

## BEGIN PROTOCOL

Now execute the Dual-Agent refinement for: **$ARGUMENTS**

Start with [ACTOR] generating the initial trajectory S₀.
```

</details>

> 💡 For the full prompt with usage examples, see [`.opencode/command/Map.md`](.opencode/command/Map.md)

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

### 🔄 Automated Visual Feedback Loop (Unity MCP Integration)

The system now supports **fully automated refinement** when Unity MCP is available. The entire loop—parameter generation, map execution, screenshot capture, and visual comparison—runs automatically:

```
┌─────────────────────────────────────────────────────────────────────┐
│              AUTOMATED VISUAL FEEDBACK LOOP (4 PHASES)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Parameter Generation (Dual-Agent)                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  /Map [reference_image.png]                                  │   │
│  │  → Actor Agent (t=0.4): Generate initial parameters          │   │
│  │  → Critic Agent (t=0.2): Validate and refine                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  Phase 2: Automated Execution (Unity MCP)                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  → Apply parameters to TileWorldCreator via MCP              │   │
│  │  → Execute map generation automatically                      │   │
│  │  → No manual intervention required                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  Phase 3: Visual Comparison (Dual-Agent)                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  → Capture screenshot via Unity MCP                          │   │
│  │  → Comparison Actor: Analyze original vs generated           │   │
│  │  → Comparison Critic: Validate similarity assessment         │   │
│  │  → Output: Similarity score (0-100%)                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  Phase 4: Auto-Refinement Decision                                  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  If similarity ≥ 80%: ✅ Complete - Output final parameters  │   │
│  │  If similarity < 80%: 🔄 Auto-refine and repeat (max 3x)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Requirements for Automated Loop

| Requirement | Description |
|-------------|-------------|
| **Unity MCP** | Must be installed and configured ([Unity MCP Setup](https://github.com/anthropics/unity-mcp)) |
| **TileWorldCreator** | Unity plugin for procedural terrain generation |
| **Unity Project** | Open with TileWorldCreator scene loaded |

#### How It Works

1. **You provide**: Reference image or text description
2. **System handles**: Parameter generation → Execution → Screenshot → Comparison → Refinement
3. **You receive**: Final optimized parameters (similarity ≥ 80% or best after 3 iterations)

#### Fallback: Manual Mode

If Unity MCP is not available, the system falls back to **manual mode**:
- Generates parameters only
- User manually applies to PCG tool
- User provides result screenshot for refinement

#### Example Usage

```bash
# Fully automated (with Unity MCP)
/Map ~/reference/mountain_village.png
# → System automatically generates, executes, compares, and refines
# → Final output: Optimized JSON parameters

# Manual fallback (without Unity MCP)
/Map ~/reference/mountain_village.png
# → System generates parameters
# → User applies manually, provides screenshot
/Map ~/screenshots/attempt1.png "refine: mountains need more height"
```

#### Similarity Threshold

| Score | Action |
|-------|--------|
| **≥ 80%** | ✅ Accept - Parameters considered optimal |
| **60-79%** | 🔄 Auto-refine - Adjust parameters and regenerate |
| **< 60%** | 🔄 Major refinement - Significant parameter changes |

The system automatically iterates up to **3 times** before returning the best result.

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
