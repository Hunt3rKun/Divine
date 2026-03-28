from divine.llm.utils.cost import CostCalculator
from divine.llm.base import TokenUsage


class TestCostCalculator:
    def test_calculate_known_model(self):
        pricing = {"gpt-4o": {"input": 2.5, "output": 10.0}}
        calc = CostCalculator(pricing)
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = calc.calculate("gpt-4o", usage)
        assert abs(cost - 0.0075) < 0.0001

    def test_calculate_unknown_model(self):
        calc = CostCalculator({})
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = calc.calculate("unknown-model", usage)
        assert cost == 0.0

    def test_update_pricing(self):
        calc = CostCalculator({})
        calc.update_pricing("new-model", {"input": 1.0, "output": 2.0})
        usage = TokenUsage(input_tokens=1000000, output_tokens=1000000, total_tokens=2000000)
        cost = calc.calculate("new-model", usage)
        assert abs(cost - 3.0) < 0.0001
