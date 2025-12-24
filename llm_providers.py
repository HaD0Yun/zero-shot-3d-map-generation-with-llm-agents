"""
llm_providers.py - LLM Provider Abstraction Layer
Based on arXiv:2512.10501

From Section 4.1:
"We employ the Claude 4.5 Sonnet model via API for inference."

This module provides:
- Abstract base class for LLM providers
- Anthropic Claude implementation (paper's choice)
- OpenAI GPT implementation (alternative)
- Mock provider for testing
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RESPONSE DATA CLASS
# =============================================================================


@dataclass
class LLMResponse:
    """
    Standardized LLM response across all providers.

    Tracks token usage for cost analysis as discussed in Section 4.3:
    "Token Usage (summed across Actor and Critic turns)"
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    finish_reason: str = "stop"
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    The architecture is tool-agnostic (Section 4.1):
    "Although our architecture is tool-agnostic, we utilize this plugin to test"

    Similarly, the LLM provider is abstracted to support multiple backends.
    """

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            system_prompt: System instructions for the model
            user_prompt: User message content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Used for context management (Section 3.3):
        "To ensure the system remains within the LLM's effective context window"
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model identifier."""
        pass


# =============================================================================
# ANTHROPIC PROVIDER (Paper's Choice)
# =============================================================================


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    From Section 4.1:
    "We employ the Claude 4.5 Sonnet model via API for inference"
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 4096,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model identifier (default: Claude 4.5 Sonnet)
            default_max_tokens: Default max tokens for generation
        """
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.default_max_tokens = default_max_tokens
        logger.info(f"Initialized Anthropic provider with model: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using Claude."""
        tokens_to_use = max_tokens if max_tokens else self.default_max_tokens
        start_time = time.time()

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens_to_use,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=message.content[0].text,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                model=self.model,
                latency_ms=latency_ms,
                finish_reason=message.stop_reason or "stop",
                raw_response=message,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        Claude uses approximately 4 characters per token for English.
        """
        return len(text) // 4

    def get_model_name(self) -> str:
        return self.model


# =============================================================================
# OPENAI PROVIDER (Alternative)
# =============================================================================


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT API provider.

    Alternative to Claude for testing generalizability.
    Supports GPT-4, GPT-4 Turbo, and other models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        default_max_tokens: int = 4096,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model identifier
            default_max_tokens: Default max tokens for generation
        """
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.default_max_tokens = default_max_tokens
        logger.info(f"Initialized OpenAI provider with model: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using GPT."""
        tokens_to_use = max_tokens if max_tokens else self.default_max_tokens
        start_time = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=tokens_to_use,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            latency_ms = (time.time() - start_time) * 1000
            choice = response.choices[0]

            return LLMResponse(
                content=choice.message.content or "",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                model=self.model,
                latency_ms=latency_ms,
                finish_reason=choice.finish_reason or "stop",
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count (GPT uses ~4 chars per token for English)."""
        return len(text) // 4

    def get_model_name(self) -> str:
        return self.model


# =============================================================================
# MOCK PROVIDER (Testing)
# =============================================================================


class MockLLMProvider(BaseLLMProvider):
    """
    Mock provider for testing without API calls.

    Useful for:
    - Unit testing
    - Development without API costs
    - Reproducible test scenarios
    """

    def __init__(self, responses: Optional[Dict[int, str]] = None):
        """
        Initialize mock provider.

        Args:
            responses: Dict mapping call number to response content
        """
        self.responses: Dict[int, str] = responses if responses is not None else {}
        self.call_count = 0
        self.call_history: list[Dict[str, Any]] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return mock response."""
        self.call_count += 1
        self.call_history.append(
            {
                "call_number": self.call_count,
                "system": system_prompt[:100],
                "user": user_prompt[:100],
                "temperature": temperature,
            }
        )

        # Return configured response or default
        content = self.responses.get(
            self.call_count,
            '{"trajectory_summary": "Mock response", "tool_plan": [{"step": 1, "objective": "Mock step", "tool_name": "MockTool", "arguments": {}, "expected_result": "Mock result"}], "risks": []}',
        )

        return LLMResponse(
            content=content,
            input_tokens=len(system_prompt + user_prompt) // 4,
            output_tokens=len(content) // 4,
            model="mock-model",
            latency_ms=10.0,
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_model_name(self) -> str:
        return "mock-model"

    def reset(self) -> None:
        """Reset call history for new test."""
        self.call_count = 0
        self.call_history = []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_provider(provider_type: str, **kwargs: Any) -> BaseLLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: One of "anthropic", "openai", "mock"
        **kwargs: Provider-specific configuration

    Returns:
        Configured LLM provider instance

    Example:
        # For paper's configuration (Claude 4.5 Sonnet)
        provider = create_provider(
            "anthropic",
            api_key="your-api-key",
            model="claude-sonnet-4-20250514"
        )
    """
    providers: Dict[str, type] = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "mock": MockLLMProvider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. Available: {list(providers.keys())}"
        )

    return providers[provider_type](**kwargs)
