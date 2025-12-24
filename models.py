"""
models.py - Complete Pydantic Models for Dual-Agent PCG System
Based on arXiv:2512.10501

This module implements all data models as specified in the paper:
- Section 3.1: Actor/Critic output structures
- Section 3.2: Prompt formulation schemas
- Section 3.3: Context management models
- Section 4.1: System configuration
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import json
import hashlib


# =============================================================================
# ENUMS
# =============================================================================


class Severity(str, Enum):
    """
    Issue severity levels as defined in the paper.

    From Section 3.1 (Table 1):
    - CRITICAL: Will definitely cause execution failure
    - MAJOR: Will likely produce incorrect results
    """

    CRITICAL = "critical"
    MAJOR = "major"


class Decision(str, Enum):
    """
    Critic decision options.

    From Section 3.2:
    "The Critic's output is a strict JSON object containing a binary decision
    ('approve' or 'revise')"
    """

    APPROVE = "approve"
    REVISE = "revise"


class TerminationReason(str, Enum):
    """
    Reasons for terminating the refinement loop (Algorithm 1).
    """

    APPROVED = "approved"  # Critic approved the trajectory
    MAX_ITERATIONS = "max_iterations"  # Reached K iterations
    ERROR = "error"  # Unrecoverable error occurred
    TIMEOUT = "timeout"  # Overall timeout exceeded


# =============================================================================
# ACTOR OUTPUT MODELS (Section 3.1, 3.2)
# =============================================================================


class ToolPlanStep(BaseModel):
    """
    A single step in the Parameter Trajectory Sequence.

    From Section 3.2 (Actor Prompt Structure):
    "Tool Plan: A granular list of sequential steps, where each step defines
    an objective, the specific tool_name, the required arguments,
    and an expected_result for verification."
    """

    step: int = Field(
        ..., ge=1, description="Step number (1-indexed, must be sequential)"
    )
    objective: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="What this specific step achieves in the PCG pipeline",
    )
    tool_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Exact tool name from API Documentation (case-sensitive)",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters with concrete values - no placeholders allowed",
    )
    expected_result: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Specific, verifiable success criteria",
    )

    @field_validator("arguments")
    @classmethod
    def validate_no_placeholders(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure no placeholder values in arguments.

        From Section 3.2:
        "The Actor is constrained to output... required arguments"
        (Must be concrete, not placeholders)
        """
        placeholders = {"TBD", "TODO", "PLACEHOLDER", "???", "N/A", ""}

        def check_value(val: Any, path: str = "") -> None:
            if isinstance(val, str):
                if val.upper() in placeholders or val.strip() == "":
                    raise ValueError(f"Placeholder value '{val}' not allowed at {path}")
            elif isinstance(val, dict):
                for k, inner_v in val.items():
                    check_value(inner_v, f"{path}.{k}")
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    check_value(item, f"{path}[{i}]")

        for key, value in v.items():
            if value is None:
                raise ValueError(f"Argument '{key}' cannot be None")
            check_value(value, key)

        return v

    def to_execution_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool execution."""
        return {"tool": self.tool_name, "arguments": self.arguments}


class ActorOutput(BaseModel):
    """
    Complete output from the Actor Agent (Parameter Trajectory Sequence S_i).

    From Section 3.1 (Table 1):
    "Output Structure: Parameter Trajectory {trajectory_summary, tool_plan, risks}"

    From Section 3.2:
    "The Actor is constrained to output a strict JSON structure comprising:
    1. Trajectory Summary: A high-level overview of the plan.
    2. Tool Plan: A granular list of sequential steps...
    3. Risk Assessment: A dedicated field for identifying potential blocking risks"
    """

    trajectory_summary: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="High-level overview explaining the overall approach",
    )
    tool_plan: List[ToolPlanStep] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Sequential list of tool invocations",
    )
    risks: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Potential blocking risks or missing information",
    )

    @model_validator(mode="after")
    def validate_step_sequence(self) -> "ActorOutput":
        """Ensure steps are sequential starting from 1."""
        expected = 1
        for step in self.tool_plan:
            if step.step != expected:
                raise ValueError(
                    f"Steps must be sequential. Expected step {expected}, got {step.step}"
                )
            expected += 1
        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        data = {
            "trajectory_summary": self.trajectory_summary,
            "tool_plan": [
                {
                    "step": s.step,
                    "objective": s.objective,
                    "tool_name": s.tool_name,
                    "arguments": s.arguments,
                    "expected_result": s.expected_result,
                }
                for s in self.tool_plan
            ],
            "risks": self.risks,
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def get_hash(self) -> str:
        """Get content hash for caching/deduplication."""
        content = self.to_json(indent=0)
        return hashlib.md5(content.encode()).hexdigest()

    @classmethod
    def from_llm_response(cls, response: str) -> "ActorOutput":
        """
        Parse LLM response into validated ActorOutput.

        Handles:
        - Raw JSON
        - JSON wrapped in ```json ... ```
        - JSON wrapped in ``` ... ```
        """
        cleaned = response.strip()

        # Handle ```json ... ``` wrapping
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # Parse JSON
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in LLM response: {e}\nResponse: {cleaned[:500]}..."
            )

        # Validate and return
        return cls.model_validate(data)

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names used."""
        return [step.tool_name for step in self.tool_plan]

    def get_step(self, step_number: int) -> Optional[ToolPlanStep]:
        """Get a specific step by number."""
        for step in self.tool_plan:
            if step.step == step_number:
                return step
        return None


# =============================================================================
# CRITIC OUTPUT MODELS (Section 3.1, 3.2)
# =============================================================================


class BlockingIssue(BaseModel):
    """
    A specific issue identified in the Actor's trajectory.

    From Section 3.1:
    "For every detected error, the Critic outputs a structured critique
    containing a description of the error and a correction suggestion"

    From Section 3.2:
    "blocking_issues or missing_information, ensuring actionable feedback"
    """

    step: int = Field(
        ..., ge=1, description="Step number where the issue was found (1-indexed)"
    )
    issue: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Clear, specific description of what is wrong",
    )
    severity: Severity = Field(
        ..., description="How severe is this issue (critical or major)"
    )
    suggestion: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Specific, actionable fix for the issue",
    )

    def to_feedback_string(self) -> str:
        """Format issue for human-readable feedback."""
        severity_marker = (
            "[CRITICAL]" if self.severity == Severity.CRITICAL else "[MAJOR]"
        )
        return (
            f"{severity_marker} Step {self.step}:\n"
            f"   Issue: {self.issue}\n"
            f"   Fix: {self.suggestion}"
        )


class CriticFeedback(BaseModel):
    """
    Complete output from the Critic Agent.

    From Section 3.1 (Table 1):
    "Output Structure: Structured Critique {decision, blocking_issues, correction_suggestion}"

    From Section 3.2:
    "The Critic's output is a strict JSON object containing a binary decision
    ('approve' or 'revise') and a list of blocking_issues or missing_information"
    """

    decision: Decision = Field(
        ..., description="Whether the trajectory is approved or needs revision"
    )
    blocking_issues: List[BlockingIssue] = Field(
        default_factory=list,
        description="List of issues that must be fixed before execution",
    )
    missing_information: List[str] = Field(
        default_factory=list, description="Unclear requirements or documentation gaps"
    )
    review_notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Notes on borderline items not flagged",
    )

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "CriticFeedback":
        """
        Ensure decision matches blocking_issues state.

        From Algorithm 1:
        "if Feedback = ∅ then return S_i"
        (Empty feedback = no blocking issues = approve)
        """
        has_issues = len(self.blocking_issues) > 0

        if has_issues and self.decision == Decision.APPROVE:
            raise ValueError(
                "Decision cannot be 'approve' when blocking_issues exist. "
                f"Found {len(self.blocking_issues)} issues."
            )
        if not has_issues and self.decision == Decision.REVISE:
            raise ValueError(
                "Decision cannot be 'revise' when no blocking_issues exist."
            )
        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_actor_feedback(self) -> str:
        """
        Format feedback for injection into Actor's revision context.

        From Section 3.3:
        "The Actor receives the Critic's feedback combined with the original
        User Prompt to generate a revised trajectory"
        """
        lines = []

        # Header
        lines.append(f"## CRITIC DECISION: {self.decision.value.upper()}")
        lines.append("")

        # Blocking issues
        if self.blocking_issues:
            lines.append("### BLOCKING ISSUES (Must Fix)")
            lines.append("")
            for issue in self.blocking_issues:
                lines.append(issue.to_feedback_string())
                lines.append("")

        # Missing information
        if self.missing_information:
            lines.append("### MISSING INFORMATION")
            for info in self.missing_information:
                lines.append(f"- {info}")
            lines.append("")

        # Review notes
        if self.review_notes:
            lines.append("### REVIEWER NOTES")
            lines.append(self.review_notes)
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_llm_response(cls, response: str) -> "CriticFeedback":
        """Parse LLM response into validated CriticFeedback."""
        cleaned = response.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in Critic response: {e}\nResponse: {cleaned[:500]}..."
            )

        return cls.model_validate(data)

    @property
    def is_approved(self) -> bool:
        """Quick check if trajectory was approved."""
        return self.decision == Decision.APPROVE

    @property
    def critical_issues(self) -> List[BlockingIssue]:
        """Get only critical severity issues."""
        return [i for i in self.blocking_issues if i.severity == Severity.CRITICAL]

    @property
    def major_issues(self) -> List[BlockingIssue]:
        """Get only major severity issues."""
        return [i for i in self.blocking_issues if i.severity == Severity.MAJOR]

    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.blocking_issues)


# =============================================================================
# ORCHESTRATION MODELS (Section 3.3, 4.1)
# =============================================================================


class IterationRecord(BaseModel):
    """Record of a single iteration in the refinement loop (Algorithm 1)."""

    iteration_number: int = Field(..., ge=0)
    trajectory: ActorOutput
    feedback: Optional[CriticFeedback] = None
    actor_duration_ms: float = Field(..., ge=0)
    critic_duration_ms: Optional[float] = Field(default=None, ge=0)
    actor_tokens_in: int = Field(default=0, ge=0)
    actor_tokens_out: int = Field(default=0, ge=0)
    critic_tokens_in: int = Field(default=0, ge=0)
    critic_tokens_out: int = Field(default=0, ge=0)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RefinementResult(BaseModel):
    """
    Complete result of the iterative refinement process.
    This is the final output from the Orchestrator (Algorithm 1).

    Output: Final Parameter Trajectory S_final
    """

    final_trajectory: ActorOutput = Field(
        ..., description="The final approved or best-effort trajectory (S_final)"
    )
    termination_reason: TerminationReason = Field(
        ..., description="Why the refinement loop terminated"
    )
    total_iterations: int = Field(
        ..., ge=1, description="Total number of iterations performed"
    )
    iteration_history: List[IterationRecord] = Field(
        default_factory=list, description="Complete history of all iterations"
    )
    total_duration_ms: float = Field(
        ..., ge=0, description="Total time taken in milliseconds"
    )
    total_input_tokens: int = Field(
        default=0, ge=0, description="Total input tokens consumed"
    )
    total_output_tokens: int = Field(
        default=0, ge=0, description="Total output tokens generated"
    )
    user_prompt: str = Field(..., description="Original user request (P_user)")
    success: bool = Field(..., description="Whether the result was approved by critic")

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def final_feedback(self) -> Optional[CriticFeedback]:
        """Get the final critic feedback (if any)."""
        if self.iteration_history:
            return self.iteration_history[-1].feedback
        return None

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        status = "[OK] SUCCESS" if self.success else "[!] BEST EFFORT"
        lines = [
            "=" * 60,
            "REFINEMENT RESULT SUMMARY",
            "=" * 60,
            f"Status: {status}",
            f"Termination: {self.termination_reason.value}",
            f"Iterations: {self.total_iterations}",
            f"Duration: {self.total_duration_ms:.2f}ms",
            f"Tokens: {self.total_tokens:,} (in: {self.total_input_tokens:,}, out: {self.total_output_tokens:,})",
            "",
            "FINAL TRAJECTORY:",
            f"  Summary: {self.final_trajectory.trajectory_summary[:100]}...",
            f"  Steps: {len(self.final_trajectory.tool_plan)}",
            f"  Tools: {', '.join(self.final_trajectory.get_tool_names())}",
        ]

        if self.final_trajectory.risks:
            lines.append(f"  Risks: {len(self.final_trajectory.risks)}")

        return "\n".join(lines)


# =============================================================================
# CONFIGURATION (Section 4.1)
# =============================================================================


class SystemConfig(BaseModel):
    """
    System configuration as specified in Section 4.1 (Experimental Setup).

    From the paper:
    "We employ the Claude 4.5 Sonnet model via API for inference.
    To balance creativity with instruction adherence, the Actor's temperature is set to 0.4.
    The Critic's temperature is set to 0.2 to ensure consistent and confident feedback.
    The maximum number of iteration cycles is set to one for all trials."
    """

    # LLM Settings (Section 4.1)
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model identifier (paper uses Claude 4.5 Sonnet)",
    )
    actor_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Actor temperature (paper: 0.4 - balance creativity and adherence)",
    )
    critic_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Critic temperature (paper: 0.2 - consistent, confident feedback)",
    )
    max_tokens: int = Field(
        default=4096, ge=256, le=16384, description="Max tokens for generation"
    )

    # Refinement Loop Settings (Section 4.1)
    max_iterations: int = Field(
        default=1, ge=1, le=10, description="Max iterations K (paper: 1 for all trials)"
    )

    # Context Management (Section 3.3)
    context_buffer_size: int = Field(
        default=16000,
        ge=1000,
        description="Total token budget for context (state-replacement strategy)",
    )

    # Timeouts (milliseconds)
    actor_timeout_ms: int = Field(
        default=60000, ge=1000, description="Actor timeout in ms"
    )
    critic_timeout_ms: int = Field(
        default=30000, ge=1000, description="Critic timeout in ms"
    )

    # Retry Settings
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Max retries per agent call"
    )
    retry_backoff_base: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Exponential backoff base"
    )

    # Observability
    enable_logging: bool = Field(default=True, description="Enable structured logging")
    log_level: str = Field(default="INFO", description="Logging level")
