---
description: "PCG map parameter generator using Dual-Agent Actor-Critic"
argument-hint: "map description"
---

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

```
1. [ACTOR] Generate initial trajectory S₀
2. [CRITIC] Review S₀ → produce feedback
3. IF issues found AND iteration < 3:
   [ACTOR] Revise trajectory based on feedback → S₁
   [CRITIC] Review S₁
   ... repeat until approved or max iterations
4. Output final approved trajectory
```

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
- `layer_heights` (list[float]): Thresholds, each in [0.0, 1.0], ascending order, length = layer_count
- `blend_factor` (float): Transition smoothness [0.0, 0.5]

#### ScatterModifier
Scatters objects on terrain.

**Required Parameters:**
- `object_type` (str): One of "rock", "tree", "grass_clump", "bush", "flower"
- `density` (float): Scatter density [0.0, 1.0]
- `valid_layers` (list[int]): Layer indices to scatter on (0-indexed)

**Optional:** `random_rotation` (bool), `scale_variation` (float), `min_distance` (float)

#### GrassDetailModifier
Adds grass coverage to a layer.

**Required Parameters:**
- `target_layer` (int): Layer index (0-indexed)
- `coverage` (float): Coverage percentage [0.0, 1.0]

**Optional:** `height_variation` (float), `color_variation` (float), `wind_response` (float)

---

## Output Format

After completing the Actor-Critic loop, output the FINAL APPROVED trajectory:

```json
{
  "protocol_log": [
    {"role": "actor", "iteration": 0, "action": "generated initial trajectory"},
    {"role": "critic", "iteration": 0, "decision": "revise|approve", "issues": []},
    ...
  ],
  "final_trajectory": {
    "trajectory_summary": "<overview of approach, 20-200 words>",
    "tool_plan": [
      {
        "step": 1,
        "objective": "<what this step achieves>",
        "tool_name": "<EXACT tool name from API docs>",
        "arguments": { ... },
        "expected_result": "<verifiable success criteria>"
      }
    ],
    "risks": ["<potential issues>"]
  },
  "termination": {
    "reason": "approved | max_iterations",
    "total_iterations": 1-3
  }
}
```

---

## Usage Examples (E)

### Example 1: Simple Island
Request: "Create a basic island"
```json
{
  "trajectory_summary": "Generate island using cellular automata with moderate fill probability for connected landmass.",
  "tool_plan": [
    {
      "step": 1,
      "objective": "Create island base shape",
      "tool_name": "CellularAutomataGenerator",
      "arguments": {
        "width": 64, "height": 64,
        "fill_probability": 0.45, "iterations": 5,
        "birth_limit": 4, "death_limit": 3
      },
      "expected_result": "Single connected landmass covering 40-50% of map"
    }
  ],
  "risks": ["Fill probability below 0.4 may create disconnected islands"]
}
```

### Example 2: Mountain with Vegetation
Request: "Mountain terrain with three elevation zones and grass on middle slopes"
```json
{
  "trajectory_summary": "Create mountain using Perlin noise, height layers for zones, grass on midlands.",
  "tool_plan": [
    {
      "step": 1,
      "objective": "Generate elevation heightmap",
      "tool_name": "PerlinNoiseGenerator",
      "arguments": {
        "width": 128, "height": 128,
        "scale": 0.05, "octaves": 4, "persistence": 0.5
      },
      "expected_result": "Smooth heightmap with natural elevation pattern"
    },
    {
      "step": 2,
      "objective": "Apply three elevation zones",
      "tool_name": "HeightLayerModifier",
      "arguments": {
        "layer_count": 3,
        "layer_heights": [0.0, 0.4, 0.75],
        "blend_factor": 0.1
      },
      "expected_result": "Three zones: lowlands (0), midlands (1), peaks (2)"
    },
    {
      "step": 3,
      "objective": "Add grass to midlands",
      "tool_name": "GrassDetailModifier",
      "arguments": {
        "target_layer": 1,
        "coverage": 0.6, "height_variation": 0.3
      },
      "expected_result": "Natural grass coverage on layer 1"
    }
  ],
  "risks": ["Perlin scale affects roughness", "Layer indices start at 0"]
}
```

---

## Token Estimation

Include approximate token counts in your output:
```json
{
  "token_estimate": {
    "actor_chars": <total chars from actor outputs>,
    "critic_chars": <total chars from critic outputs>,
    "estimated_tokens": <total_chars / 4>
  }
}
```

---

## BEGIN PROTOCOL

Now execute the Dual-Agent refinement for: **$ARGUMENTS**

Start with [ACTOR] generating the initial trajectory S₀.
