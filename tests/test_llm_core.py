import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from divine.llm.catalog import DEFAULT_MODELS, list_models
from divine.llm.config import LLMSettings
from divine.llm.types import LLMRequest, Message, TokenUsage


class LLMCoreTests(unittest.TestCase):
    def test_default_models_include_required_provider_families(self):
        self.assertEqual(DEFAULT_MODELS["openai"], "gpt-5.5")
        self.assertEqual(DEFAULT_MODELS["anthropic"], "claude-opus-4-1-20250805")
        self.assertEqual(DEFAULT_MODELS["deepseek"], "deepseek-v4-flash")
        self.assertEqual(DEFAULT_MODELS["dashscope"], "qwen3-max-2026-01-23")
        self.assertEqual(DEFAULT_MODELS["zhipu"], "glm-5.1")

    def test_list_models_can_filter_provider(self):
        models = list_models("deepseek")
        self.assertEqual([item.model for item in models][:2], ["deepseek-v4-flash", "deepseek-v4-pro"])

    def test_settings_from_file_resolves_provider_specific_values(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "llm.json"
            path.write_text(
                """{
  "provider": "deepseek",
  "generation": {"max_tokens": 2048},
  "providers": {
    "deepseek": {
      "api_key": "test-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-v4-pro"
    }
  }
}
""",
                encoding="utf-8",
            )
            settings = LLMSettings.from_file(path)

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")
        self.assertEqual(settings.default_max_tokens, 2048)

    def test_dashscope_uses_native_sdk_without_compatible_base_url_default(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "llm.json"
            path.write_text(
                """{
  "provider": "dashscope",
  "providers": {
    "dashscope": {
      "api_key": "test-key"
    }
  }
}
""",
                encoding="utf-8",
            )
            settings = LLMSettings.from_file(path)

        self.assertIsNone(settings.base_url)

    def test_provider_override_keeps_config_file_as_secret_source(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "llm.json"
            path.write_text(
                """{
  "provider": "openai",
  "providers": {
    "deepseek": {
      "api_key": "test-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-v4-pro"
            }
  }
}
""",
                encoding="utf-8",
            )
            settings = LLMSettings.from_file(path, provider="deepseek")

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-v4-pro")
        self.assertEqual(settings.api_key, "test-key")

    def test_request_normalizes_message_dataclasses_and_dicts(self):
        request = LLMRequest(
            messages=[
                Message("user", "hello"),
                {"role": "assistant", "content": "world"},
            ]
        )
        self.assertEqual(
            request.normalized_messages(),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )

    def test_token_usage_normalizes_openai_style_usage(self):
        usage = TokenUsage.from_raw(
            {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        )
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.total_tokens, 19)
        self.assertEqual(usage.cached_tokens, 5)
        self.assertEqual(usage.reasoning_tokens, 3)

    def test_token_usage_normalizes_anthropic_style_usage(self):
        usage = TokenUsage.from_raw(
            {
                "input_tokens": 10,
                "output_tokens": 8,
                "cache_read_input_tokens": 4,
            }
        )
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 8)
        self.assertEqual(usage.total_tokens, 18)
        self.assertEqual(usage.cached_tokens, 4)

    def test_token_usage_normalizes_deepseek_cache_usage(self):
        usage = TokenUsage.from_raw(
            {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "prompt_cache_hit_tokens": 12,
                "prompt_cache_miss_tokens": 8,
                "total_tokens": 25,
            }
        )
        self.assertEqual(usage.cached_tokens, 12)
        self.assertEqual(usage.cache_miss_tokens, 8)


if __name__ == "__main__":
    unittest.main()
