from divine.tools.adapter import ToolAdapter, ToolResult, _extract_title


def test_tool_result_succeeded_only_for_success_status():
    assert ToolResult("x", "success", {}).succeeded is True
    assert ToolResult("x", "failed", {}).succeeded is False


def test_extract_title_from_html():
    assert _extract_title("<html><title> Divine Lab </title></html>") == "Divine Lab"
    assert _extract_title("<html><body>No title</body></html>") is None


def test_https_probe_normalizes_url_scheme(monkeypatch):
    called = {}

    def fake_http_request(tool_name, url, *, timeout):
        called["tool_name"] = tool_name
        called["url"] = url
        called["timeout"] = timeout
        return ToolResult(tool_name, "success", {"url": url})

    adapter = ToolAdapter()
    monkeypatch.setattr(adapter, "_http_request", fake_http_request)

    result = adapter.https_probe("example.com", timeout=1.5)

    assert result.succeeded is True
    assert called == {
        "tool_name": "https_probe",
        "url": "https://example.com",
        "timeout": 1.5,
    }


def test_host_info_returns_structured_platform_data():
    result = ToolAdapter().host_info()

    assert result.succeeded is True
    assert result.artifact_type == "host_info"
    assert result.output["system"]
    assert result.output["platform"]

