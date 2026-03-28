import pytest
from divine.codeact.sandbox import Sandbox, ExecutionResult


class TestExecutionResult:
    def test_success_result(self):
        r = ExecutionResult(success=True, stdout="hello", execution_time=0.5)
        assert r.success
        assert r.stdout == "hello"

    def test_failure_result(self):
        r = ExecutionResult(success=False, stderr="error")
        assert not r.success


class TestSandbox:
    async def test_execute_simple_code(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("x = 1 + 1")
        assert result.success

    async def test_execute_with_stdout(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    async def test_execute_returns_last_expression(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("1 + 1")
        assert result.success
        assert result.return_value == 2

    async def test_execute_error(self):
        sandbox = Sandbox(timeout=10)
        result = await sandbox.execute("raise ValueError('test error')")
        assert not result.success
        assert "test error" in result.stderr

    async def test_persistent_namespace(self):
        sandbox = Sandbox(timeout=10)
        await sandbox.execute("my_var = 42")
        result = await sandbox.execute("print(my_var)")
        assert result.success
        assert "42" in result.stdout

    async def test_reset_clears_namespace(self):
        sandbox = Sandbox(timeout=10)
        await sandbox.execute("my_var = 42")
        sandbox.reset()
        result = await sandbox.execute("print(my_var)")
        assert not result.success

    async def test_setup_injects_stdlib(self):
        sandbox = Sandbox(timeout=10)
        sandbox.setup({"add": lambda a, b: a + b})
        result = await sandbox.execute("result = add(3, 4)\nprint(result)")
        assert result.success
        assert "7" in result.stdout


class TestStdlib:
    def test_create_stdlib(self):
        from divine.blackboard import Blackboard
        from divine.codeact.stdlib import create_stdlib

        bb = Blackboard()
        stdlib = create_stdlib(bb)
        assert "run_command" in stdlib
        assert "http_request" in stdlib
        assert "bb_read" in stdlib
        assert "bb_write" in stdlib

    def test_run_command(self):
        from divine.codeact.stdlib import run_command

        result = run_command("echo hello")
        assert "hello" in result["stdout"]
        assert result["returncode"] == 0
