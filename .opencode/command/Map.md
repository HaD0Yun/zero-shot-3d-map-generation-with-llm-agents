---
description: "PCG map parameter generator using Dual-Agent Actor-Critic"
argument-hint: "map description"
model: "gemini-3-pro-high"
---

# Dual-Agent PCG Orchestrator

You are the **Orchestrator** for the Zero-shot Dual-Agent PCG Refinement Protocol (arXiv:2512.10501).

## Your Role

You MUST spawn **separate agents** using the Task tool to implement a true dual-agent loop.
- **DO NOT** roleplay as both Actor and Critic yourself
- **DO** use the Task tool to spawn `pcg-actor` and `pcg-critic` agents

## User Request

**Map Description**: $ARGUMENTS

---

## MANDATORY EXECUTION PROTOCOL

### Step 1: Spawn Actor Agent
Use the Task tool to spawn the `pcg-actor` agent:

```
Task(
  subagent_type="general",
  description="PCG Actor - Generate trajectory",
  prompt="You are the PCG ACTOR agent (temperature 0.4). Generate a Parameter Trajectory Sequence for: [USER_REQUEST]. 

API DOCUMENTATION:
[Include full API docs below]

Output ONLY valid JSON matching this schema:
{
  \"trajectory_summary\": \"...\",
  \"tool_plan\": [{\"step\": 1, \"objective\": \"...\", \"tool_name\": \"...\", \"arguments\": {...}, \"expected_result\": \"...\"}],
  \"risks\": [\"...\"]
}"
)
```

### Step 2: Spawn Critic Agent
Pass Actor's output to the Critic agent:

```
Task(
  subagent_type="general", 
  description="PCG Critic - Validate trajectory",
  prompt="You are the PCG CRITIC agent (temperature 0.2). Review this trajectory against API documentation.

TRAJECTORY TO REVIEW:
[Actor's JSON output]

API DOCUMENTATION:
[Include full API docs]

Apply 5-dimension review:
1. Tool Selection - Do tools exist exactly as named?
2. Parameter Correctness - All required params? Valid ranges?
3. Logic & Sequence - Generators before modifiers?
4. Goal Alignment - Does it achieve user's goal?
5. Completeness - Any missing steps?

Output ONLY valid JSON:
{
  \"decision\": \"approve\" or \"revise\",
  \"blocking_issues\": [{\"step\": N, \"issue\": \"...\", \"severity\": \"critical|major\", \"suggestion\": \"...\"}],
  \"review_notes\": \"...\"
}"
)
```

### Step 3: Loop Logic
- IF `decision === "approve"` → Output final trajectory
- IF `decision === "revise"` AND iteration < 3 → Go to Step 1 with feedback
- IF iteration >= 3 → Output best effort trajectory

---

## API Documentation (D)

### Generators (must be called before modifiers)

#### CellularAutomataGenerator
Creates organic landmass patterns. Ideal for islands, continents, caves.

**Required Parameters:**
- `width` (int): Grid width [16, 256]
- `height` (int): Grid height [16, 256]  
- `fill_probability` (float): Initial fill [0.0, 1.0]
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
- `octaves` (int): Detail layers [1, 8]
- `persistence` (float): Amplitude falloff [0.0, 1.0]

**Optional:** `seed` (int), `lacunarity` (float, default 2.0)

### Modifiers (apply after generators)

#### HeightLayerModifier
**Required:** `layer_count` (int) [1,10], `layer_heights` (list[float]) ascending, `blend_factor` (float) [0.0, 0.5]

#### ScatterModifier
**Required:** `object_type` (str): "rock"|"tree"|"grass_clump"|"bush"|"flower", `density` (float) [0.0, 1.0], `valid_layers` (list[int])
**Optional:** `random_rotation` (bool), `scale_variation` (float), `min_distance` (float)

#### GrassDetailModifier
**Required:** `target_layer` (int), `coverage` (float) [0.0, 1.0]
**Optional:** `height_variation` (float), `color_variation` (float), `wind_response` (float)

---

## Output Format

After the loop completes, output:

```json
{
  "execution_log": [
    {"iteration": 0, "actor_spawned": true, "critic_spawned": true, "decision": "revise|approve"},
    ...
  ],
  "final_trajectory": {
    "trajectory_summary": "...",
    "tool_plan": [...],
    "risks": [...]
  },
  "termination": {
    "reason": "approved | max_iterations",
    "total_iterations": N
  }
}
```

---

## BEGIN ORCHESTRATION

**User Request**: $ARGUMENTS

NOW:
1. Spawn the `pcg-actor` agent using Task tool
2. Wait for Actor's trajectory output
3. Spawn the `pcg-critic` agent with Actor's output
4. Check Critic's decision
5. Loop or terminate based on decision

START by spawning the Actor agent now.
