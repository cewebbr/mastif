"""Startup checks for model, judge, and Mind2Web dependencies."""

import json
import os
from typing import Any, Dict, Iterable

from adapters import AnthropicAdapter, HuggingFaceAdapter, OllamaAdapter, OpenAIAdapter


class PreflightError(RuntimeError):
    """Raised when an experiment dependency is unavailable before startup."""


class PreflightChecker:
    """Verify configured external services before an experiment starts."""

    def __init__(self, config):
        self.config = config

    def run(self):
        models = self.config.get("models", []) or []
        mode = self.config.get("test_mode", "standard")

        print("\n🔎 Running preflight checks...")
        for model_entry in models:
            model_name = self._model_name(model_entry)
            self._check_model(model_name)
            self._check_tokenizer(model_name, self._tokenizer_id(model_entry))

        if mode == "mind2web":
            self._check_dataset()
            judge_model = self.config.get("mind2web_judge_model") or self.config.get(
                "judge_model", "gpt-4o-mini"
            )
            self._check_judge(judge_model)

        print("✅️ Preflight checks passed.\n")

    def _check_tokenizer(self, model_name: str, tokenizer_id: str = None):
        """Validate the configured or inferred tokenizer used for token metrics."""
        provider = self._tokenizer_provider(model_name)
        if tokenizer_id:
            self._load_huggingface_tokenizer(tokenizer_id)
            print(f"  ✅️ Tokenizer available: {tokenizer_id} (configured)")
            return

        if provider == "openai":
            import tiktoken

            try:
                tiktoken.encoding_for_model(model_name)
            except KeyError:
                tiktoken.get_encoding("cl100k_base")
                print(f"  ⚠️ Tokenizer for '{model_name}' is unknown; using cl100k_base approximation.")
                return
            print(f"  ✅️ Tokenizer available for model: {model_name}")
            return

        if provider in {"anthropic", "ollama"}:
            import tiktoken

            tiktoken.get_encoding("cl100k_base")
            print(f"  ✅️ Tokenizer available: cl100k_base approximation ({provider})")
            return

        try:
            self._load_huggingface_tokenizer(model_name)
            print(f"  ✅️ Tokenizer available: {model_name}")
        except PreflightError:
            # The result exporter already falls back to cl100k_base if an
            # inferred Hugging Face tokenizer cannot be loaded.
            import tiktoken

            tiktoken.get_encoding("cl100k_base")
            print(f"  ⚠️ Hugging Face tokenizer unavailable for '{model_name}'; token metrics will use cl100k_base approximation.")

    @staticmethod
    def _load_huggingface_tokenizer(tokenizer_id: str):
        from transformers import AutoTokenizer

        last_error = None
        for options in ({}, {"use_fast": False}):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **options)
                if tokenizer.encode("preflight tokenizer check"):
                    return
                last_error = ValueError("tokenizer returned no token IDs")
            except Exception as error:
                last_error = error
        raise PreflightError(
            f"Tokenizer '{tokenizer_id}' is unavailable: {last_error}"
        ) from last_error

    @staticmethod
    def _tokenizer_provider(model_name: str) -> str:
        lower = model_name.lower()
        if any(lower.startswith(prefix) for prefix in ("gpt-", "text-", "openai")):
            return "openai"
        if lower.startswith("claude-"):
            return "anthropic"
        if ":" in model_name and "/" not in model_name:
            return "ollama"
        if "/" in model_name:
            return "huggingface"
        return "unknown"

    def _check_model(self, model_name: str):
        adapter = self._create_adapter(model_name)
        response = self._generate(adapter, "Reply with exactly: OK")
        if not response.strip() or self._looks_like_error(response):
            raise PreflightError(
                f"Model '{model_name}' is unavailable: {self._clean(response)}"
            )
        print(f"  ✅️ Model configured: {model_name}")

    def _check_judge(self, model_name: str):
        adapter = self._create_adapter(model_name)
        prompt = (
            'Return only this exact compact JSON, with no other text: '
            '{"task_understanding":{"score":1},"task_adherence":{"score":1},'
            '"task_completion":{"score":1}}'
        )
        response = self._generate(adapter, prompt, max_tokens=256)
        if self._looks_like_error(response):
            raise PreflightError(
                f"Judge '{model_name}' is unavailable: {self._clean(response)}"
            )
        parsed = self._parse_json_object(response)
        if parsed is None:
            raise PreflightError(
                f"Judge '{model_name}' returned invalid JSON: {self._clean(response)}"
            )
        required = {"task_understanding", "task_adherence", "task_completion"}
        if not required.issubset(parsed.keys()):
            missing = ", ".join(sorted(required - set(parsed)))
            raise PreflightError(
                f"Judge '{model_name}' returned JSON missing: {missing}"
            )
        print(f"  ✅️ Judge available: {model_name}")

    @staticmethod
    def _parse_json_object(response: str):
        """Parse a JSON object, accepting fenced or surrounded judge output."""
        cleaned = response.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

    def _check_dataset(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise PreflightError(
                "Mind2Web requires HF_TOKEN to verify dataset access."
            )
        try:
            from datasets import load_dataset

            stream = load_dataset(
                "osunlp/Mind2Web",
                split="train",
                streaming=True,
                token=token,
            )
            next(iter(stream))
        except Exception as error:
            raise PreflightError(
                f"Mind2Web dataset is unavailable: {error}"
            ) from error
        print("  ✅️ Mind2Web dataset available: osunlp/Mind2Web")

    def _create_adapter(self, model_name: str):
        if model_name.startswith("gpt-") or model_name.startswith("openai/"):
            self._require("OPENAI_API_KEY", model_name)
            return OpenAIAdapter(model_name, api_key=os.getenv("OPENAI_API_KEY"))
        if model_name.startswith("claude-") or model_name.startswith("anthropic/"):
            self._require("ANTHROPIC_API_KEY", model_name)
            return AnthropicAdapter(model_name, api_key=os.getenv("ANTHROPIC_API_KEY"))
        if ":" in model_name and "/" not in model_name:
            return OllamaAdapter(model_name)
        self._require("HF_TOKEN", model_name)
        return HuggingFaceAdapter(model_name, api_key=os.getenv("HF_TOKEN"))

    def _generate(self, adapter, prompt: str, max_tokens: int = 32) -> str:
        try:
            return adapter.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        except Exception as error:
            raise PreflightError(
                f"{adapter.model_name} request failed: {error}"
            ) from error

    @staticmethod
    def _model_name(entry: Any) -> str:
        if isinstance(entry, dict):
            return entry["id"]
        return str(entry)

    @staticmethod
    def _tokenizer_id(entry: Any) -> str:
        return entry.get("tokenizer") if isinstance(entry, dict) else None

    @staticmethod
    def _require(variable: str, model_name: str):
        if not os.getenv(variable):
            raise PreflightError(
                f"{variable} is required for model '{model_name}'."
            )

    @staticmethod
    def _looks_like_error(response: str) -> bool:
        lower = response.strip().lower()
        return lower.startswith(("error:", "api error:", "anthropic error:"))

    @staticmethod
    def _clean(response: str) -> str:
        return response.strip()[:300] or "empty response"