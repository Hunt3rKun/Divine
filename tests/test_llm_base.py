from divine.llm.base import LLMMessage, LLMResponse, TokenUsage


class TestLLMMessage:
    def test_create_message(self):
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_system_message(self):
        msg = LLMMessage(role="system", content="You are a pentester")
        assert msg.role == "system"


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.total_tokens == 0

    def test_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150


class TestLLMResponse:
    def test_create_response(self):
        resp = LLMResponse(
            content="Hello",
            model="gpt-4o",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert resp.content == "Hello"
        assert resp.cost == 0.0
