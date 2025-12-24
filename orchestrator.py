"""
orchestrator.py - Core Orchestrator Implementation
Based on arXiv:2512.10501

This module implements:
- Algorithm 1: Zero-shot Dual-Agent PCG Refinement
- ContextManager: State-replacement strategy (Section 3.3)
- ActorAgent: Semantic interpreter (Section 3.1)
- CriticAgent: Static verifier (Section 3.1)
- Orchestrator: Main controller

Paper Reference (Algorithm 1):
    Input: P_user, D, E, K
    Output: S_final

    Context_actor ← {P_user, D, E}
    S₀ ← Actor(Context_actor)
    i ← 0

    while i < K do:
        Feedback ← Critic(Sᵢ, D, E)
        if Feedback = ∅ then return Sᵢ
        Context_actor ← UpdateContext(Sᵢ, Feedback)
        Sᵢ₊₁ ← Actor(Context_actor)
        i ← i + 1

    return Sₖ
"""

from __future__ import annotations
import asyncio
import time
import logging
from typing import List, Optional, Tuple, Any

from .models import (
    ActorOutput,
    CriticFeedback,
    SystemConfig,
    RefinementResult,
    IterationRecord,
    TerminationReason,
)
from .llm_providers import BaseLLMProvider, LLMResponse
from .prompts import ACTOR_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT MANAGER (Section 3.3 - State-Replacement Strategy)
# =============================================================================


class ContextManager:
    """
    Implements the State-Replacement Strategy from Section 3.3.

    From the paper:
    "To ensure the system remains within the LLM's effective context window
    across multiple iterations, we implement a state-replacement strategy.
    Rather than appending the entire history of the dialogue, the system
    maintains a fixed-size context buffer. The context is updated in-place:
    the distinct 'Initial Trajectory' block is overwritten by the
    'Revised Trajectory' at the start of each new evaluation cycle."

    Key features:
    - Fixed-size context buffer (NOT appending full history)
    - "Initial Trajectory" → "Revised Trajectory" overwrite
    - Preserves: current best hypothesis + latest critique
    - Discards: obsolete trajectories
    """

    def __init__(
        self,
        api_documentation: str,
        usage_examples: List[str],
        llm_provider: BaseLLMProvider,
        context_buffer_size: int = 16000,
    ):
        """
        Initialize context manager.

        Args:
            api_documentation: PCG tool documentation (D)
            usage_examples: Validated usage examples (E)
            llm_provider: LLM provider for token counting
            context_buffer_size: Max tokens for context
        """
        self.api_documentation = api_documentation
        self.usage_examples = usage_examples
        self.llm_provider = llm_provider
        self.context_buffer_size = context_buffer_size

        # Current state (replaced, not appended)
        self._current_trajectory: Optional[ActorOutput] = None
        self._current_feedback: Optional[CriticFeedback] = None
        self._iteration: int = 0

        logger.debug("ContextManager initialized")

    def build_actor_prompt(self, user_prompt: str) -> str:
        """
        Build context for Actor agent.

        From Algorithm 1:
        Context_actor ← {P_user, D, E}

        Implements state-replacement: only current trajectory + feedback included,
        not full history of all previous attempts.
        """
        sections = []

        # P_user: User request
        sections.append("## USER REQUEST")
        sections.append(user_prompt)
        sections.append("")

        # D: API Documentation
        sections.append("## API DOCUMENTATION")
        sections.append(self.api_documentation)
        sections.append("")

        # E: Usage Examples
        if self.usage_examples:
            sections.append("## USAGE EXAMPLES")
            for i, example in enumerate(self.usage_examples, 1):
                sections.append(f"### Example {i}")
                sections.append(example)
                sections.append("")

        # Revision context (if this is a revision) - STATE REPLACEMENT
        # From Section 3.3: "the distinct 'Initial Trajectory' block is overwritten
        # by the 'Revised Trajectory'"
        if self._current_trajectory and self._current_feedback:
            sections.append("## REVISION CONTEXT")
            sections.append("")
            sections.append(
                "Your previous trajectory received feedback and must be revised."
            )
            sections.append("")
            sections.append("### YOUR PREVIOUS TRAJECTORY")
            sections.append("```json")
            sections.append(self._current_trajectory.to_json())
            sections.append("```")
            sections.append("")
            sections.append("### CRITIC FEEDBACK")
            sections.append(self._current_feedback.to_actor_feedback())
            sections.append("")
            sections.append(
                "Please generate a REVISED trajectory that addresses ALL blocking issues."
            )

        return "\n".join(sections)

    def build_critic_prompt(self, trajectory: ActorOutput, user_prompt: str) -> str:
        """
        Build context for Critic agent.

        From Algorithm 1:
        Feedback ← Critic(Sᵢ, D, E)
        """
        sections = []

        # Trajectory to review (Sᵢ)
        sections.append("## TRAJECTORY TO REVIEW")
        sections.append("```json")
        sections.append(trajectory.to_json())
        sections.append("```")
        sections.append("")

        # Original user request (for goal alignment check)
        sections.append("## ORIGINAL USER REQUEST")
        sections.append(user_prompt)
        sections.append("")

        # D: API Documentation (source of truth)
        sections.append("## API DOCUMENTATION (Source of Truth)")
        sections.append(self.api_documentation)
        sections.append("")

        # E: Usage Examples
        if self.usage_examples:
            sections.append("## VALIDATED USAGE EXAMPLES")
            for i, example in enumerate(self.usage_examples, 1):
                sections.append(f"### Example {i}")
                sections.append(example)
                sections.append("")

        # Task instruction
        sections.append("## YOUR TASK")
        sections.append(
            "Review the trajectory against the API Documentation and user request. "
            "Apply the 5-dimension review framework and provide your verdict."
        )

        return "\n".join(sections)

    def update(self, trajectory: ActorOutput, feedback: CriticFeedback) -> None:
        """
        Update context with new trajectory and feedback.

        From Section 3.3:
        "This preserves the most relevant state information—the current best
        hypothesis and the latest critique—while discarding obsolete trajectories"

        This REPLACES the previous state (state-replacement strategy),
        rather than appending to history.
        """
        self._current_trajectory = trajectory
        self._current_feedback = feedback
        self._iteration += 1

        logger.debug(f"Context updated for iteration {self._iteration}")

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._iteration

    @property
    def has_previous_attempt(self) -> bool:
        """Whether there's a previous trajectory to revise."""
        return self._current_trajectory is not None

    def reset(self) -> None:
        """Reset state for a new request."""
        self._current_trajectory = None
        self._current_feedback = None
        self._iteration = 0
        logger.debug("Context manager reset")


# =============================================================================
# ACTOR AGENT (Section 3.1)
# =============================================================================


class ActorAgent:
    """
    Actor Agent (Semantic Interpreter).

    From Section 3.1:
    "The Actor functions as the semantic interpreter and generator.
    Given a user's natural language prompt (P_user) describing the desired map,
    the Actor parses the intent and synthesizes an initial Parameter Trajectory
    Sequence (S₀). This sequence maps high-level, open-ended design concepts
    into specific PCG algorithm steps and their corresponding low-level
    parameter values."

    From Section 4.1:
    "To balance creativity with instruction adherence, the Actor's temperature
    is set to 0.4"
    """

    def __init__(self, llm_provider: BaseLLMProvider, config: SystemConfig):
        """
        Initialize Actor agent.

        Args:
            llm_provider: LLM provider for generation
            config: System configuration with temperature settings
        """
        self.llm = llm_provider
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ActorAgent")

    async def invoke(self, user_prompt: str) -> Tuple[ActorOutput, LLMResponse]:
        """
        Generate a Parameter Trajectory Sequence.

        From Algorithm 1:
        S₀ ← Actor(Context_actor)
        Sᵢ₊₁ ← Actor(Context_actor)

        Args:
            user_prompt: Built context with user request, docs, examples, feedback

        Returns:
            Tuple of (validated ActorOutput, raw LLMResponse)
        """
        self.logger.info("Actor generating trajectory")

        # Call LLM with paper's temperature setting (0.4)
        response = await self.llm.generate(
            system_prompt=ACTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.actor_temperature,
            max_tokens=self.config.max_tokens,
        )

        # Parse and validate response
        try:
            output = ActorOutput.from_llm_response(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse Actor response: {e}")
            self.logger.debug(f"Raw response: {response.content[:500]}...")
            raise ValueError(f"Actor output parsing failed: {e}")

        self.logger.info(
            f"Actor generated trajectory with {len(output.tool_plan)} steps "
            f"(tokens: in={response.input_tokens}, out={response.output_tokens})"
        )

        return output, response


# =============================================================================
# CRITIC AGENT (Section 3.1)
# =============================================================================


class CriticAgent:
    """
    Critic Agent (Static Verifier).

    From Section 3.1:
    "The generated trajectory (S₀) is passed to the Critic agent, which acts
    as a static verifier. The Critic evaluates the proposed parameters against
    two provided knowledge sources:
    1. API Documentation: Formal definitions of available PCG algorithms...
    2. Reference Demonstration: Validated example of algorithm orchestration..."

    From Section 3.2:
    "Crucially, the Critic is instructed to adopt a conservative review policy:
    it must only flag 'blocking issues' if it is absolutely certain of an error"

    From Section 4.1:
    "The Critic's temperature is set to 0.2 to ensure consistent and confident
    feedback"
    """

    def __init__(self, llm_provider: BaseLLMProvider, config: SystemConfig):
        """
        Initialize Critic agent.

        Args:
            llm_provider: LLM provider for evaluation
            config: System configuration with temperature settings
        """
        self.llm = llm_provider
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CriticAgent")

    async def invoke(self, user_prompt: str) -> Tuple[CriticFeedback, LLMResponse]:
        """
        Evaluate a Parameter Trajectory Sequence.

        From Algorithm 1:
        Feedback ← Critic(Sᵢ, D, E)

        Args:
            user_prompt: Built context with trajectory, docs, examples

        Returns:
            Tuple of (validated CriticFeedback, raw LLMResponse)
        """
        self.logger.info("Critic evaluating trajectory")

        # Call LLM with paper's temperature setting (0.2)
        response = await self.llm.generate(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.critic_temperature,
            max_tokens=self.config.max_tokens // 2,  # Critic needs less output
        )

        # Parse and validate response
        try:
            feedback = CriticFeedback.from_llm_response(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse Critic response: {e}")
            self.logger.debug(f"Raw response: {response.content[:500]}...")
            raise ValueError(f"Critic output parsing failed: {e}")

        self.logger.info(
            f"Critic decision: {feedback.decision.value} "
            f"({feedback.issue_count} issues, "
            f"tokens: in={response.input_tokens}, out={response.output_tokens})"
        )

        return feedback, response


# =============================================================================
# ORCHESTRATOR (Algorithm 1)
# =============================================================================


class Orchestrator:
    """
    Main orchestrator implementing Algorithm 1 from the paper.

    From Section 3.3:
    "The core of our architecture is an internal dialogic feedback loop that
    allows the system to resolve ambiguities and correct hallucinations
    progressively. The Actor receives the Critic's feedback combined with
    the original User Prompt to generate a revised trajectory (Sᵢ₊₁)"

    Algorithm 1: Zero-shot Dual-Agent PCG Refinement
    ================================================
    Input: P_user, D, E, K (max iterations)
    Output: S_final

    Context_actor ← {P_user, D, E}
    S₀ ← Actor(Context_actor)  // Initial generation
    i ← 0

    while i < K do:
        Feedback ← Critic(Sᵢ, D, E)

        if Feedback = ∅ then         // Valid configuration found
            return Sᵢ

        Context_actor ← UpdateContext(Sᵢ, Feedback)
        Sᵢ₊₁ ← Actor(Context_actor)
        i ← i + 1

    return Sₖ                        // Best effort if max iterations reached
    """

    def __init__(
        self,
        config: SystemConfig,
        llm_provider: BaseLLMProvider,
        api_documentation: str,
        usage_examples: List[str],
    ):
        """
        Initialize orchestrator.

        Args:
            config: System configuration
            llm_provider: LLM provider for both agents
            api_documentation: PCG tool documentation (D)
            usage_examples: Validated usage examples (E)
        """
        self.config = config
        self.llm_provider = llm_provider

        # Initialize components
        self.context_manager = ContextManager(
            api_documentation=api_documentation,
            usage_examples=usage_examples,
            llm_provider=llm_provider,
            context_buffer_size=config.context_buffer_size,
        )
        self.actor = ActorAgent(llm_provider, config)
        self.critic = CriticAgent(llm_provider, config)

        self.logger = logging.getLogger(f"{__name__}.Orchestrator")

        # Metrics tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def execute(self, user_prompt: str) -> RefinementResult:
        """
        Execute the full iterative refinement protocol (Algorithm 1).

        From the paper:
        "Through successive iterations, the trajectory becomes increasingly
        constrained by the functional requirements provided by the Critic,
        ensuring convergence toward a valid execution plan without requiring
        gradient updates."

        Args:
            user_prompt: Natural language description of desired PCG output (P_user)

        Returns:
            RefinementResult with final trajectory (S_final) and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting refinement for: {user_prompt[:100]}...")

        # Reset state
        self.context_manager.reset()
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        iteration_history: List[IterationRecord] = []
        current_trajectory: Optional[ActorOutput] = None

        try:
            # ============================================================
            # Step 1: Initial generation (S₀)
            # From Algorithm 1: S₀ ← Actor(Context_actor)
            # ============================================================
            actor_start = time.time()
            actor_prompt = self.context_manager.build_actor_prompt(user_prompt)
            current_trajectory, actor_response = await self._invoke_actor_with_retry(
                actor_prompt
            )
            actor_duration = (time.time() - actor_start) * 1000

            # Track tokens
            self._total_input_tokens += actor_response.input_tokens
            self._total_output_tokens += actor_response.output_tokens

            # ============================================================
            # Step 2: Iterative refinement loop
            # From Algorithm 1: while i < K do
            # ============================================================
            for i in range(self.config.max_iterations):
                self.logger.info(f"Iteration {i + 1}/{self.config.max_iterations}")

                # --------------------------------------------------------
                # Critic evaluation
                # From Algorithm 1: Feedback ← Critic(Sᵢ, D, E)
                # --------------------------------------------------------
                critic_start = time.time()
                critic_prompt = self.context_manager.build_critic_prompt(
                    current_trajectory, user_prompt
                )
                feedback, critic_response = await self._invoke_critic_with_retry(
                    critic_prompt
                )
                critic_duration = (time.time() - critic_start) * 1000

                # Track tokens
                self._total_input_tokens += critic_response.input_tokens
                self._total_output_tokens += critic_response.output_tokens

                # Record iteration
                iteration_history.append(
                    IterationRecord(
                        iteration_number=i,
                        trajectory=current_trajectory,
                        feedback=feedback,
                        actor_duration_ms=actor_duration,
                        critic_duration_ms=critic_duration,
                        actor_tokens_in=actor_response.input_tokens,
                        actor_tokens_out=actor_response.output_tokens,
                        critic_tokens_in=critic_response.input_tokens,
                        critic_tokens_out=critic_response.output_tokens,
                    )
                )

                # --------------------------------------------------------
                # Check termination: Critic approved
                # From Algorithm 1: if Feedback = ∅ then return Sᵢ
                # --------------------------------------------------------
                if feedback.is_approved:
                    self.logger.info("[OK] Critic APPROVED trajectory")

                    return RefinementResult(
                        final_trajectory=current_trajectory,
                        termination_reason=TerminationReason.APPROVED,
                        total_iterations=i + 1,
                        iteration_history=iteration_history,
                        total_duration_ms=(time.time() - start_time) * 1000,
                        total_input_tokens=self._total_input_tokens,
                        total_output_tokens=self._total_output_tokens,
                        user_prompt=user_prompt,
                        success=True,
                    )

                # Check if this is the last iteration
                if i >= self.config.max_iterations - 1:
                    break

                # --------------------------------------------------------
                # Update context (state replacement)
                # From Algorithm 1: Context_actor ← UpdateContext(Sᵢ, Feedback)
                # --------------------------------------------------------
                self.context_manager.update(current_trajectory, feedback)

                # --------------------------------------------------------
                # Generate revised trajectory
                # From Algorithm 1: Sᵢ₊₁ ← Actor(Context_actor)
                # --------------------------------------------------------
                self.logger.info(
                    f"Revising trajectory ({feedback.issue_count} issues to address)"
                )
                actor_start = time.time()
                actor_prompt = self.context_manager.build_actor_prompt(user_prompt)
                (
                    current_trajectory,
                    actor_response,
                ) = await self._invoke_actor_with_retry(actor_prompt)
                actor_duration = (time.time() - actor_start) * 1000

                # Track tokens
                self._total_input_tokens += actor_response.input_tokens
                self._total_output_tokens += actor_response.output_tokens

            # ============================================================
            # Max iterations reached - return best effort
            # From Algorithm 1: return Sₖ
            # ============================================================
            self.logger.warning(
                f"Max iterations ({self.config.max_iterations}) reached - "
                "returning best effort"
            )

            return RefinementResult(
                final_trajectory=current_trajectory,
                termination_reason=TerminationReason.MAX_ITERATIONS,
                total_iterations=self.config.max_iterations,
                iteration_history=iteration_history,
                total_duration_ms=(time.time() - start_time) * 1000,
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                user_prompt=user_prompt,
                success=False,
            )

        except asyncio.TimeoutError:
            self.logger.error("Overall timeout exceeded")

            return RefinementResult(
                final_trajectory=current_trajectory
                if current_trajectory
                else self._create_empty_trajectory(),
                termination_reason=TerminationReason.TIMEOUT,
                total_iterations=len(iteration_history),
                iteration_history=iteration_history,
                total_duration_ms=(time.time() - start_time) * 1000,
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                user_prompt=user_prompt,
                success=False,
            )

        except Exception as e:
            self.logger.error(f"Unrecoverable error: {e}")
            raise

    async def _invoke_actor_with_retry(
        self, prompt: str
    ) -> Tuple[ActorOutput, LLMResponse]:
        """Invoke Actor with retry logic and exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self.actor.invoke(prompt),
                    timeout=self.config.actor_timeout_ms / 1000,
                )

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Actor timeout after {self.config.actor_timeout_ms}ms"
                )
                self.logger.warning(f"Actor timeout (attempt {attempt + 1})")

            except ValueError as e:
                # JSON parsing error - retry might help
                last_error = e
                self.logger.warning(f"Actor parse error (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = e
                self.logger.warning(f"Actor error (attempt {attempt + 1}): {e}")

            # Exponential backoff before retry
            if attempt < self.config.max_retries - 1:
                backoff = self.config.retry_backoff_base**attempt
                self.logger.debug(f"Backing off for {backoff}s")
                await asyncio.sleep(backoff)

        raise (
            last_error if last_error else RuntimeError("Actor failed after all retries")
        )

    async def _invoke_critic_with_retry(
        self, prompt: str
    ) -> Tuple[CriticFeedback, LLMResponse]:
        """Invoke Critic with retry logic and exponential backoff."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self.critic.invoke(prompt),
                    timeout=self.config.critic_timeout_ms / 1000,
                )

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Critic timeout after {self.config.critic_timeout_ms}ms"
                )
                self.logger.warning(f"Critic timeout (attempt {attempt + 1})")

            except ValueError as e:
                last_error = e
                self.logger.warning(f"Critic parse error (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = e
                self.logger.warning(f"Critic error (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                backoff = self.config.retry_backoff_base**attempt
                await asyncio.sleep(backoff)

        raise (
            last_error
            if last_error
            else RuntimeError("Critic failed after all retries")
        )

    def _create_empty_trajectory(self) -> ActorOutput:
        """Create an empty trajectory for error cases."""
        from .models import ToolPlanStep

        return ActorOutput(
            trajectory_summary="Error: No trajectory generated due to system error",
            tool_plan=[
                ToolPlanStep(
                    step=1,
                    objective="Placeholder due to error",
                    tool_name="ErrorPlaceholder",
                    arguments={"error": True},
                    expected_result="None - trajectory generation failed",
                )
            ],
            risks=["Trajectory generation failed - this is a placeholder"],
        )
