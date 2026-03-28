import pytest
from divine.llm.utils.retry import RetryHandler, RetryConfig


class TestRetryHandler:
    async def test_success_first_try(self):
        handler = RetryHandler(RetryConfig())
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await handler.execute(fn)
        assert result == "ok"
        assert call_count == 1

    async def test_retry_on_retryable_error(self):
        handler = RetryHandler(RetryConfig(max_attempts=3, base_delay=0.01))
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "ok"

        result = await handler.execute(fn)
        assert result == "ok"
        assert call_count == 3

    async def test_max_attempts_exceeded(self):
        handler = RetryHandler(RetryConfig(max_attempts=2, base_delay=0.01))

        async def fn():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await handler.execute(fn)

    def test_delay_calculation(self):
        handler = RetryHandler(RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0))
        d1 = handler._calculate_delay(1)
        d2 = handler._calculate_delay(2)
        d3 = handler._calculate_delay(3)
        assert d1 >= 0.5  # with jitter
        assert d2 >= 1.0
        assert d3 <= 10.0
