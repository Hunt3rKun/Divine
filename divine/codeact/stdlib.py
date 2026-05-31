import base64
import subprocess
from urllib.parse import urlparse

from divine.blackboard.blackboard import Blackboard
from divine.tools import ToolAdapter, ToolResult


def run_command(cmd: str, timeout: int = 60) -> dict:
    """执行 shell 命令"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def http_request(
    url: str,
    method: str = "GET",
    headers: dict = None,
    data: str = None,
    timeout: int = 30,
) -> dict:
    """发送 HTTP 请求"""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, method=method, headers=headers or {})
        if data:
            req.data = data.encode()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
            return {"status": resp.status, "headers": dict(resp.headers), "body": body}
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "headers": dict(e.headers),
            "body": e.read().decode(errors="replace"),
        }
    except Exception as e:
        return {"status": 0, "headers": {}, "body": "", "error": str(e)}


def parse_nmap(output: str) -> list[dict]:
    """简单解析 nmap 输出"""
    results = []
    for line in output.splitlines():
        line = line.strip()
        if "/tcp" in line or "/udp" in line:
            parts = line.split()
            if len(parts) >= 3:
                port_proto = parts[0]
                port, proto = port_proto.split("/")
                results.append(
                    {
                        "port": int(port),
                        "protocol": proto,
                        "state": parts[1],
                        "service": parts[2] if len(parts) > 2 else "",
                        "version": " ".join(parts[3:]) if len(parts) > 3 else "",
                    }
                )
    return results


def parse_url(url: str) -> dict:
    """解析 URL"""
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname or "",
        "port": parsed.port,
        "path": parsed.path,
        "query": parsed.query,
    }


def b64encode(data: str) -> str:
    return base64.b64encode(data.encode()).decode()


def b64decode(data: str) -> str:
    return base64.b64decode(data.encode()).decode()


def _tool_result_to_dict(result: ToolResult) -> dict:
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "input": dict(result.input),
        "output": dict(result.output),
        "error": result.error,
        "artifact_type": result.artifact_type,
        "succeeded": result.succeeded,
    }


def create_stdlib(blackboard: Blackboard) -> dict:
    """创建注入 sandbox 的标准库"""
    tools = ToolAdapter()
    return {
        "run_command": run_command,
        "http_request": http_request,
        "tcp_connect_check": lambda host, port, timeout=2.0: _tool_result_to_dict(
            tools.tcp_connect_check(host, port, timeout=timeout)
        ),
        "http_probe": lambda url, timeout=5.0: _tool_result_to_dict(
            tools.http_probe(url, timeout=timeout)
        ),
        "https_probe": lambda url, timeout=5.0: _tool_result_to_dict(
            tools.https_probe(url, timeout=timeout)
        ),
        "path_probe": lambda base_url, paths=None, timeout=5.0: _tool_result_to_dict(
            tools.path_probe(base_url, paths=paths, timeout=timeout)
        ),
        "host_info": lambda: _tool_result_to_dict(tools.host_info()),
        "bb_read": blackboard.read,
        "bb_write": blackboard.write,
        "parse_nmap": parse_nmap,
        "parse_url": parse_url,
        "b64encode": b64encode,
        "b64decode": b64decode,
    }
