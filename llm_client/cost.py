"""
费用计算器
"""
from typing import Dict, Optional

from config.config import DEFAULT_PRICING
from .models import TokenUsage



class CostCalculator:
    """
    根据模型价格表计算费用
    价格表结构: {model_name: {"input": price, "output": price}}
    价格单位: $/1M tokens
    """

    def __init__(self, pricing: Optional[Dict[str, Dict[str, float]]] = None):
        self._pricing = pricing or DEFAULT_PRICING.copy()

    def calculate(self, model: str, usage: TokenUsage) -> float:
        """
        计算费用

        Args:
            model: 模型名称
            usage: token 使用量

        Returns:
            费用（人民币）
        """
        price = self._get_price(model)
        if price is None:
            return 0.0

        # 价格单位是 ￥/1M tokens，需要除以 1,000,000
        input_cost = (usage.input_tokens / 1_000_000) * price["input"]
        output_cost = (usage.output_tokens / 1_000_000) * price["output"]

        return round(input_cost + output_cost, 6)

    def _get_price(self, model: str) -> Optional[Dict[str, float]]:
        """
        获取模型价格，支持模糊匹配
        例如：gpt-4o-2024-05-13 可以匹配 gpt-4o
        """
        # 精确匹配
        if model in self._pricing:
            return self._pricing[model]

        # 前缀匹配（去掉日期后缀等）
        for known_model in self._pricing:
            if model.startswith(known_model) or known_model.startswith(model):
                return self._pricing[known_model]

        return None

    def update_pricing(self, model: str, input_price: float, output_price: float):
        """更新单个模型的价格"""
        self._pricing[model] = {"input": input_price, "output": output_price}

    def set_pricing(self, pricing: Dict[str, Dict[str, float]]):
        """替换整个价格表"""
        self._pricing = pricing.copy()

    def get_pricing(self) -> Dict[str, Dict[str, float]]:
        """获取当前价格表"""
        return self._pricing.copy()