# -*- coding: utf-8 -*-
"""LLM wrapper supporting multiple backends."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from collections import deque


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    backend: str = "mock"  # mock, openai, anthropic, local
    model_name: str = "mock-model"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMWrapper:
    """Unified LLM interface with pluggable backends."""

    def __init__(self, config: Optional[LLMConfig] = None, max_history: int = 1000):
        self.config = config or LLMConfig()
        self.call_history: deque = deque(maxlen=max_history)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM."""
        self.call_history.append({"prompt": prompt, "system": system_prompt})

        if self.config.backend == "mock":
            return self._mock_generate(prompt, system_prompt)
        elif self.config.backend == "openai":
            return self._openai_generate(prompt, system_prompt)
        elif self.config.backend == "anthropic":
            return self._anthropic_generate(prompt, system_prompt)
        elif self.config.backend == "local":
            return self._local_generate(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def _mock_generate(self, prompt: str, system_prompt: str) -> str:
        """Mock backend for testing without API keys."""
        if "hypothesis" in prompt.lower():
            return "Hypothesis: Elevated Abeta oligomer concentration triggers microglial M1 activation via NF-kB pathway, leading to neuroinflammation-driven tau hyperphosphorylation."
        elif "experiment" in prompt.lower():
            return "Experiment: Sweep Abeta production rate [0.05-0.5] while monitoring microglial activation state transitions and tau phosphorylation levels over 1000 timesteps."
        elif "analyze" in prompt.lower() or "result" in prompt.lower():
            return "Analysis: Simulation results show a nonlinear relationship between Abeta oligomer levels and microglial activation threshold at ~0.3 concentration units."
        return f"Mock response to: {prompt[:80]}..."

    def _openai_generate(self, prompt: str, system_prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=self.config.model_name, messages=messages,
                temperature=self.config.temperature, max_tokens=self.config.max_tokens
            )
            return resp.choices[0].message.content
        except ImportError as e:
            import logging
            logging.warning(f"OpenAI not installed, falling back to mock: {e}")
            return self._mock_generate(prompt, system_prompt)
        except Exception as e:
            import logging
            logging.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def _anthropic_generate(self, prompt: str, system_prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.api_key)
            resp = client.messages.create(
                model=self.config.model_name, max_tokens=self.config.max_tokens,
                system=system_prompt or "You are a scientific research assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        except ImportError as e:
            import logging
            logging.warning(f"Anthropic not installed, falling back to mock: {e}")
            return self._mock_generate(prompt, system_prompt)
        except Exception as e:
            import logging
            logging.error(f"Anthropic API call failed: {e}")
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    def _local_generate(self, prompt: str, system_prompt: str) -> str:
        try:
            import requests
            url = self.config.base_url or "http://localhost:8080/v1/completions"
            resp = requests.post(url, json={"prompt": prompt, "max_tokens": self.config.max_tokens}, timeout=30)
            resp.raise_for_status()
            return resp.json().get("choices", [{}])[0].get("text", "")
        except ImportError as e:
            import logging
            logging.warning(f"Requests not installed, falling back to mock: {e}")
            return self._mock_generate(prompt, system_prompt)
        except Exception as e:
            import logging
            logging.error(f"Local LLM call failed: {e}")
            raise RuntimeError(f"Local LLM call failed: {e}") from e

    def get_stats(self) -> Dict:
        return {"backend": self.config.backend, "total_calls": len(self.call_history)}
