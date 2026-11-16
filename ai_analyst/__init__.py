"""AI-асистент продуктового аналітика: pandas рахує, LLM інтерпретує, verify перевіряє числа."""
from .assistant import AIAnalyst
from .llm import LLMError, get_backend
from .profiling import load_csv

__all__ = ["AIAnalyst", "LLMError", "get_backend", "load_csv"]
