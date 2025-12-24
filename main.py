"""
main.py - Complete Runnable Example
Based on arXiv:2512.10501

"Zero-shot 3D Map Generation with LLM Agents:
A Dual-Agent Architecture for Procedural Content Generation"

This script demonstrates the complete dual-agent PCG system with:
- Sample API documentation (TileWorldCreator-style)
- Usage examples for reference
- Full execution pipeline

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY=your-key-here

    # Run the example
    python -m dual_agent_pcg.main
"""

import asyncio
import os
import logging
import sys
from typing import List

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_agent_pcg.models import SystemConfig
from dual_agent_pcg.llm_providers import create_provider
from dual_agent_pcg.orchestrator import Orchestrator


# =============================================================================
# SAMPLE API DOCUMENTATION
# =============================================================================
# This mimics the TileWorldCreator plugin documentation referenced in Section 4.1

API_DOCUMENTATION = """
# TileWorldCreator PCG Tools API

This documentation defines the available Procedural Content Generation (PCG) tools
for terrain and map generation. All tools must be used exactly as specified.

## Generators

Generators create the base terrain data. They must be called before modifiers.

### CellularAutomataGenerator
Creates organic landmass patterns using cellular automata algorithm.
Ideal for creating natural-looking islands, continents, or cave systems.

**Required Parameters:**
- width (int): Grid width in tiles. Valid range: [16, 256]
- height (int): Grid height in tiles. Valid range: [16, 256]
- fill_probability (float): Initial cell fill probability. Valid range: [0.0, 1.0]
  - Lower values (0.3-0.4): Create smaller, more scattered landmasses
  - Medium values (0.45-0.55): Create balanced, connected landmasses
  - Higher values (0.6-0.7): Create larger, more solid landmasses
- iterations (int): Number of CA simulation iterations. Valid range: [1, 10]
  - More iterations = smoother, more organic shapes
- birth_limit (int): Neighbor count threshold for cell birth. Valid range: [0, 8]
  - Typically 4 for natural terrain
- death_limit (int): Neighbor count threshold for cell survival. Valid range: [0, 8]
  - Typically 3 for natural terrain

**Optional Parameters:**
- seed (int): Random seed for reproducibility. Default: random

**Output:** Binary grid where 1 = land, 0 = empty

### PerlinNoiseGenerator
Creates smooth, continuous terrain heightmaps using Perlin noise.
Ideal for natural elevation patterns, rolling hills, and mountain ranges.

**Required Parameters:**
- width (int): Grid width. Valid range: [16, 512]
- height (int): Grid height. Valid range: [16, 512]
- scale (float): Noise scale (lower = smoother). Valid range: [0.01, 1.0]
  - 0.01-0.03: Very smooth, large features
  - 0.04-0.08: Medium detail, good for mountains
  - 0.1+: High detail, rough terrain
- octaves (int): Number of detail layers. Valid range: [1, 8]
  - More octaves = more fine detail
- persistence (float): Amplitude falloff per octave. Valid range: [0.0, 1.0]
  - 0.3-0.5: Smooth transitions
  - 0.5-0.7: Balanced detail
  - 0.7+: Rough, detailed terrain

**Optional Parameters:**
- seed (int): Random seed. Default: random
- lacunarity (float): Frequency multiplier per octave. Default: 2.0

**Output:** Heightmap with values in [0.0, 1.0]

## Modifiers

Modifiers transform existing terrain data. They must be applied after generators.

### HeightLayerModifier
Applies discrete height layers to terrain, creating stepped elevation zones.
Essential for creating distinct biomes or gameplay areas.

**Required Parameters:**
- layer_count (int): Number of height layers. Valid range: [1, 10]
- layer_heights (list[float]): Height thresholds for each layer. 
  - Each value must be in [0.0, 1.0]
  - Must be sorted in ascending order
  - Length must equal layer_count
- blend_factor (float): Transition smoothness between layers. Valid range: [0.0, 0.5]
  - 0.0: Sharp transitions
  - 0.1-0.2: Subtle blending
  - 0.3+: Smooth gradients

**Output:** Terrain with discrete layer assignments

### ScatterModifier
Scatters objects (rocks, trees, etc.) across terrain based on rules.
Use for environmental decoration and detail.

**Required Parameters:**
- object_type (str): Type of object to scatter.
  - Valid values: "rock", "tree", "grass_clump", "bush", "flower"
- density (float): Scatter density. Valid range: [0.0, 1.0]
  - 0.1-0.2: Sparse scattering
  - 0.3-0.5: Medium density
  - 0.6+: Dense coverage
- valid_layers (list[int]): Which height layers to scatter on (0-indexed)
  - Must reference valid layer indices

**Optional Parameters:**
- random_rotation (bool): Randomize object rotation. Default: true
- scale_variation (float): Random scale variation. Valid range: [0.0, 1.0]. Default: 0.0
- min_distance (float): Minimum distance between objects. Default: 0.0

**Output:** Scatter map with object placements

### GrassDetailModifier
Adds grass/vegetation details to specific terrain layers.
Provides fine detail for ground coverage.

**Required Parameters:**
- target_layer (int): Height layer to apply grass (0-indexed)
- coverage (float): Grass coverage percentage. Valid range: [0.0, 1.0]

**Optional Parameters:**
- height_variation (float): Grass blade height randomness. Valid range: [0.0, 1.0]. Default: 0.2
- color_variation (float): Color variation amount. Valid range: [0.0, 1.0]. Default: 0.1
- wind_response (float): How much grass responds to wind. Valid range: [0.0, 1.0]. Default: 0.5

**Output:** Grass detail layer for the specified terrain layer

## Execution Order

For correct results, follow this execution order:
1. Generators (create base terrain)
2. HeightLayerModifier (if using layers)
3. ScatterModifier / GrassDetailModifier (decoration)

Applying modifiers before generators will result in errors.
"""


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
# From Section 3.1: "Reference Demonstration: Validated example of algorithm orchestration"

USAGE_EXAMPLES: List[str] = [
    """
### Example: Simple Island Generation

User Request: "Create a basic island"

```json
{
  "trajectory_summary": "Generate a basic island using cellular automata with moderate fill probability to create a single connected landmass with organic coastline. The parameters are tuned to produce approximately 45-50% land coverage with smooth edges.",
  "tool_plan": [
    {
      "step": 1,
      "objective": "Create island base shape using cellular automata",
      "tool_name": "CellularAutomataGenerator",
      "arguments": {
        "width": 64,
        "height": 64,
        "fill_probability": 0.45,
        "iterations": 5,
        "birth_limit": 4,
        "death_limit": 3
      },
      "expected_result": "Single connected landmass covering approximately 40-50% of map area with natural-looking coastline"
    }
  ],
  "risks": [
    "Fill probability below 0.4 may create disconnected islands instead of single landmass",
    "Random seed not specified - results will vary between runs"
  ]
}
```
""",
    """
### Example: Multi-Layer Mountain with Vegetation

User Request: "Create a mountain terrain with three elevation zones and grass on the middle slopes"

```json
{
  "trajectory_summary": "Create a mountain terrain with three elevation zones using Perlin noise for natural heightmap, height layers for distinct zones (lowlands, midlands, peaks), and grass coverage on the middle elevation layer. The noise scale is set low for large, smooth mountain features.",
  "tool_plan": [
    {
      "step": 1,
      "objective": "Generate natural elevation heightmap using Perlin noise",
      "tool_name": "PerlinNoiseGenerator",
      "arguments": {
        "width": 128,
        "height": 128,
        "scale": 0.05,
        "octaves": 4,
        "persistence": 0.5
      },
      "expected_result": "Smooth heightmap with natural-looking mountain elevation pattern, values ranging 0.0-1.0"
    },
    {
      "step": 2,
      "objective": "Apply three distinct elevation zones",
      "tool_name": "HeightLayerModifier",
      "arguments": {
        "layer_count": 3,
        "layer_heights": [0.0, 0.4, 0.75],
        "blend_factor": 0.1
      },
      "expected_result": "Three distinct zones: lowlands (0-40% height), midlands (40-75%), peaks (75-100%)"
    },
    {
      "step": 3,
      "objective": "Add grass coverage to middle elevation zone",
      "tool_name": "GrassDetailModifier",
      "arguments": {
        "target_layer": 1,
        "coverage": 0.6,
        "height_variation": 0.3
      },
      "expected_result": "Natural grass coverage on midland areas (layer 1) with height variation for visual interest"
    }
  ],
  "risks": [
    "Perlin scale value significantly affects terrain roughness - may need adjustment for desired look",
    "Layer height thresholds should be tuned based on desired zone proportions",
    "Grass on layer 1 assumes layer indices start at 0 - verify layer assignment"
  ]
}
```
""",
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================


async def main():
    """
    Run the complete dual-agent PCG system.

    This implements the full pipeline from arXiv:2512.10501:
    1. Initialize system with paper's configuration (Section 4.1)
    2. Execute Algorithm 1 (Section 3.3)
    3. Display results
    """

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("=" * 60)
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("=" * 60)
        print("\nTo use this system, you need an Anthropic API key.")
        print("\nSet it with:")
        print("  export ANTHROPIC_API_KEY=your-key-here  (Linux/Mac)")
        print("  set ANTHROPIC_API_KEY=your-key-here     (Windows)")
        print("\nAlternatively, you can run in mock mode for testing:")
        print("  python -m dual_agent_pcg.main --mock")
        return

    # ========================================================================
    # Create configuration (Section 4.1 - System Configuration)
    # ========================================================================
    # From the paper:
    # - Actor temperature: 0.4 (balance creativity and adherence)
    # - Critic temperature: 0.2 (consistent, confident feedback)
    # - Max iterations: 1 (for all trials)
    # - LLM: Claude 4.5 Sonnet

    config = SystemConfig(
        llm_model="claude-sonnet-4-20250514",
        actor_temperature=0.4,  # Paper: "balance creativity with instruction adherence"
        critic_temperature=0.2,  # Paper: "consistent and confident feedback"
        max_iterations=1,  # Paper: "set to one for all trials"
        max_tokens=4096,
        actor_timeout_ms=60000,
        critic_timeout_ms=30000,
        max_retries=3,
    )

    # ========================================================================
    # Create LLM provider
    # ========================================================================
    llm_provider = create_provider(
        provider_type="anthropic", api_key=api_key, model=config.llm_model
    )

    # ========================================================================
    # Create orchestrator
    # ========================================================================
    orchestrator = Orchestrator(
        config=config,
        llm_provider=llm_provider,
        api_documentation=API_DOCUMENTATION,
        usage_examples=USAGE_EXAMPLES,
    )

    # ========================================================================
    # User prompt
    # ========================================================================
    # From Section 4.1 (Experiment I - Complex Task Protocol):
    # "generating a complex 3D mountain map subject to four specific constraints:
    # - Terrain: Formation of a single mountain peak
    # - Morphology: The mountain must consist of exactly three height layers
    # - Detailing: Grass spots must be applied specifically to the peak layer
    # - Scatter: Rocks must be scattered in areas not occupied by the mountain"

    user_prompt = """
Create a mountainous terrain for a fantasy RPG game with the following requirements:

1. A single connected mountain range (not multiple disconnected islands)
2. Three distinct height levels: lowlands, midlands, and peaks
3. Grass coverage on the middle elevation layer (midlands)
4. Scattered rocks in the lowland areas around the mountain base

The terrain should look natural and be suitable for exploration gameplay.
    """.strip()

    # ========================================================================
    # Execute
    # ========================================================================
    print("=" * 70)
    print("DUAL-AGENT PCG SYSTEM")
    print("Based on arXiv:2512.10501")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - LLM Model: {config.llm_model}")
    print(f"  - Actor Temperature: {config.actor_temperature}")
    print(f"  - Critic Temperature: {config.critic_temperature}")
    print(f"  - Max Iterations: {config.max_iterations}")
    print(f"\nUser Prompt:\n{user_prompt}")
    print("-" * 70)
    print("\nExecuting Algorithm 1 (Zero-shot Dual-Agent PCG Refinement)...")
    print()

    # Execute the refinement
    result = await orchestrator.execute(user_prompt)

    # ========================================================================
    # Display results
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    status = "[OK] SUCCESS" if result.success else "[!] BEST EFFORT"
    print(f"\n{status}")
    print(f"Termination Reason: {result.termination_reason.value}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Duration: {result.total_duration_ms:.2f}ms")
    print(
        f"Tokens Used: {result.total_tokens:,} (in: {result.total_input_tokens:,}, out: {result.total_output_tokens:,})"
    )

    print("\n" + "-" * 70)
    print("FINAL TRAJECTORY (S_final)")
    print("-" * 70)

    print(f"\nSummary:\n{result.final_trajectory.trajectory_summary}")

    print(f"\nTool Plan ({len(result.final_trajectory.tool_plan)} steps):")
    for step in result.final_trajectory.tool_plan:
        print(f"\n  Step {step.step}: {step.objective}")
        print(f"    Tool: {step.tool_name}")
        print(f"    Arguments: {step.arguments}")
        print(f"    Expected: {step.expected_result}")

    if result.final_trajectory.risks:
        print(f"\nRisks Identified ({len(result.final_trajectory.risks)}):")
        for risk in result.final_trajectory.risks:
            print(f"  - {risk}")

    # Show iteration history if multiple iterations
    if len(result.iteration_history) > 0:
        print("\n" + "-" * 70)
        print("ITERATION HISTORY")
        print("-" * 70)

        for record in result.iteration_history:
            decision = record.feedback.decision.value if record.feedback else "N/A"
            issues = record.feedback.issue_count if record.feedback else 0
            print(f"\n  Iteration {record.iteration_number + 1}:")
            print(f"    Actor Duration: {record.actor_duration_ms:.2f}ms")
            print(
                f"    Critic Duration: {record.critic_duration_ms:.2f}ms"
                if record.critic_duration_ms
                else ""
            )
            print(f"    Critic Decision: {decision}")
            print(f"    Issues Found: {issues}")

            if record.feedback and record.feedback.blocking_issues:
                print("    Issues:")
                for issue in record.feedback.blocking_issues:
                    print(
                        f"      [{issue.severity.value}] Step {issue.step}: {issue.issue}"
                    )

    print("\n" + "=" * 70)
    print("END OF RESULTS")
    print("=" * 70)

    # Return result for programmatic use
    return result


async def main_mock():
    """Run with mock provider for testing without API calls."""
    from dual_agent_pcg.llm_providers import MockLLMProvider

    print("=" * 70)
    print("DUAL-AGENT PCG SYSTEM (MOCK MODE)")
    print("Based on arXiv:2512.10501")
    print("=" * 70)
    print("\nRunning in mock mode - no actual API calls will be made.")

    # Mock responses
    mock_actor_response = """{
  "trajectory_summary": "Generate a mountain terrain using cellular automata for base landmass, Perlin noise for elevation, height layers for distinct zones, grass on midlands, and rocks in lowlands.",
  "tool_plan": [
    {
      "step": 1,
      "objective": "Create single connected mountain base",
      "tool_name": "CellularAutomataGenerator",
      "arguments": {
        "width": 128,
        "height": 128,
        "fill_probability": 0.48,
        "iterations": 6,
        "birth_limit": 4,
        "death_limit": 3
      },
      "expected_result": "Single connected landmass covering 45-55% of map"
    },
    {
      "step": 2,
      "objective": "Apply elevation using Perlin noise",
      "tool_name": "PerlinNoiseGenerator",
      "arguments": {
        "width": 128,
        "height": 128,
        "scale": 0.04,
        "octaves": 5,
        "persistence": 0.55
      },
      "expected_result": "Smooth elevation gradient for mountain terrain"
    },
    {
      "step": 3,
      "objective": "Create three height layers",
      "tool_name": "HeightLayerModifier",
      "arguments": {
        "layer_count": 3,
        "layer_heights": [0.0, 0.35, 0.7],
        "blend_factor": 0.12
      },
      "expected_result": "Three distinct zones: lowlands, midlands, peaks"
    },
    {
      "step": 4,
      "objective": "Add grass to midlands",
      "tool_name": "GrassDetailModifier",
      "arguments": {
        "target_layer": 1,
        "coverage": 0.65,
        "height_variation": 0.3
      },
      "expected_result": "Natural grass coverage on middle elevation"
    },
    {
      "step": 5,
      "objective": "Scatter rocks in lowlands",
      "tool_name": "ScatterModifier",
      "arguments": {
        "object_type": "rock",
        "density": 0.15,
        "valid_layers": [0],
        "random_rotation": true
      },
      "expected_result": "Rocks scattered around mountain base"
    }
  ],
  "risks": [
    "Cellular automata fill_probability may need tuning for desired landmass size",
    "Height layer thresholds affect zone proportions"
  ]
}"""

    mock_critic_response = """{
  "decision": "approve",
  "blocking_issues": [],
  "missing_information": [],
  "review_notes": "All tool names exist, parameters are within valid ranges, and the execution sequence is correct. The trajectory addresses all user requirements."
}"""

    mock_provider = MockLLMProvider(
        responses={1: mock_actor_response, 2: mock_critic_response}
    )

    config = SystemConfig(
        actor_temperature=0.4, critic_temperature=0.2, max_iterations=1
    )

    orchestrator = Orchestrator(
        config=config,
        llm_provider=mock_provider,
        api_documentation=API_DOCUMENTATION,
        usage_examples=USAGE_EXAMPLES,
    )

    user_prompt = "Create a mountain terrain with three height levels, grass on midlands, and rocks in lowlands."

    print(f"\nUser Prompt: {user_prompt}\n")
    print("-" * 70)

    result = await orchestrator.execute(user_prompt)

    print(f"\n{'[OK] SUCCESS' if result.success else '[!] BEST EFFORT'}")
    print(f"Termination: {result.termination_reason.value}")
    print(f"Iterations: {result.total_iterations}")

    print(f"\nFinal Trajectory Summary:")
    print(f"  {result.final_trajectory.trajectory_summary[:200]}...")
    print(f"\nSteps: {len(result.final_trajectory.tool_plan)}")
    for step in result.final_trajectory.tool_plan:
        print(f"  {step.step}. {step.tool_name}")

    return result


if __name__ == "__main__":
    import sys

    if "--mock" in sys.argv:
        asyncio.run(main_mock())
    else:
        asyncio.run(main())
