"""LLM abstraction layer: backend ABC and utilities."""

from react_review.llm.base import LLMBackend, parse_llm_response

__all__ = ["LLMBackend", "parse_llm_response"]
