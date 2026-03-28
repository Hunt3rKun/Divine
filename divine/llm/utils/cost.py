from divine.llm.base import TokenUsage


class CostCalculator:
    def __init__(self, pricing: dict[str, dict[str, float]] = None):
        self._pricing = pricing or {}

    def calculate(self, model: str, usage: TokenUsage) -> float:
        price = self._get_price(model)
        if not price:
            return 0.0
        input_cost = (usage.input_tokens / 1_000_000) * price.get("input", 0)
        output_cost = (usage.output_tokens / 1_000_000) * price.get("output", 0)
        return input_cost + output_cost

    def update_pricing(self, model: str, prices: dict[str, float]) -> None:
        self._pricing[model] = prices

    def _get_price(self, model: str) -> dict[str, float] | None:
        if model in self._pricing:
            return self._pricing[model]
        for key in self._pricing:
            if model.startswith(key):
                return self._pricing[key]
        return None
