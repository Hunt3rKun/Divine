from divine.llm.providers.openai import OpenAIProvider
from divine.llm.providers.anthropic import AnthropicProvider
from divine.llm.providers.zhipu import ZhipuProvider
from divine.llm.providers.minimax import MiniMaxProvider
from divine.llm.providers.openai_compat import OpenAICompatProvider

PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "zhipu": ZhipuProvider,
    "minimax": MiniMaxProvider,
    "openai_compat": OpenAICompatProvider,
}
