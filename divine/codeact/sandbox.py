import ast
import asyncio
import io
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time: float = 0.0


class Sandbox:
    def __init__(self, timeout: int = 60):
        self._timeout = timeout
        self._globals: dict = {"__builtins__": __builtins__}

    def setup(self, stdlib: dict) -> None:
        self._globals.update(stdlib)

    async def execute(self, code: str) -> ExecutionResult:
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync, code),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                stderr=f"Timeout: execution exceeded {self._timeout}s",
            )

    def _execute_sync(self, code: str) -> ExecutionResult:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        return_value = None
        start_time = time.time()
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            tree = ast.parse(code)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(last_expr)
            if tree.body:
                compiled = compile(tree, "<sandbox>", "exec")
                exec(compiled, self._globals)
            if last_expr:
                compiled_expr = compile(last_expr, "<sandbox>", "eval")
                return_value = eval(compiled_expr, self._globals)
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                return_value=return_value,
                execution_time=execution_time,
            )
        except Exception:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=traceback.format_exc(),
                execution_time=execution_time,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def reset(self) -> None:
        self._globals = {"__builtins__": __builtins__}
