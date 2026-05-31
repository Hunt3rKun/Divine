from divine.context import CachePolicy, ContextBuilder, ContextSection, ConversationMemory, PromptSegment, TokenBudget


def _segments():
    static = [
        PromptSegment(
            name="agent.rules",
            content="你是授权渗透测试框架中的规划智能体。输出必须是 JSON。",
            section=ContextSection.STATIC,
            stable=True,
            cache_policy="explicit",
        )
    ]
    mission = [
        PromptSegment(
            name="mission.scope",
            content="目标仅限本地靶场，禁止公网未授权操作。",
            section=ContextSection.MISSION,
            stable=True,
            cache_policy="explicit",
        )
    ]
    working = [
        PromptSegment(
            name="blackboard.summary",
            content="DAG 当前有 3 个待执行节点。",
            section=ContextSection.WORKING,
        )
    ]
    current = [
        PromptSegment(
            name="planner.next",
            content="请选择下一个可执行节点。",
            section=ContextSection.CURRENT,
        )
    ]
    return static, mission, working, current


def test_openai_context_builder_emits_cache_key_and_stable_system_prefix():
    static, mission, working, current = _segments()
    result = ContextBuilder(
        provider="openai",
        agent="planner",
        cache_policy=CachePolicy(retention="24h"),
    ).build_request(
        static_segments=static,
        mission_segments=mission,
        working_segments=working,
        current_segments=current,
        trace_id="trace-openai",
    )

    assert result.request.system.startswith("## agent.rules@v1")
    assert "blackboard.summary" not in result.request.system
    assert result.request.extra["prompt_cache_key"].startswith("divine:planner:")
    assert result.request.extra["prompt_cache_retention"] == "24h"
    assert result.request.prompt_trace["stable_segment_names"] == ["agent.rules", "mission.scope"]
    assert "Working Context" in result.request.messages[0].content


def test_anthropic_context_builder_marks_stable_prefix_cache_breakpoint():
    static, mission, working, current = _segments()
    result = ContextBuilder(
        provider="anthropic",
        agent="planner",
        cache_policy=CachePolicy(ttl="1h"),
    ).build_request(
        static_segments=static,
        mission_segments=mission,
        working_segments=working,
        current_segments=current,
    )

    assert isinstance(result.request.system, list)
    assert result.request.system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "blackboard.summary" not in result.request.system[-1]["text"]
    assert "blackboard.summary" in result.request.messages[0].content


def test_deepseek_context_builder_keeps_prefix_stable_without_provider_extra():
    static, mission, working, current = _segments()
    result = ContextBuilder(provider="deepseek", agent="planner").build_request(
        static_segments=static,
        mission_segments=mission,
        working_segments=working,
        current_segments=current,
    )

    assert result.request.extra == {}
    assert result.cache_hint.cache_key.startswith("divine:planner:")
    assert result.request.system == "\n\n".join(segment.render() for segment in [*static, *mission])


def test_token_budget_trims_dynamic_context_but_preserves_stable_prefix():
    static, mission, _, current = _segments()
    working = [
        PromptSegment(
            name="large.context",
            content="x" * 5000,
            section=ContextSection.WORKING,
        )
    ]
    result = ContextBuilder(
        provider="openai",
        agent="planner",
        token_budget=TokenBudget(max_input_tokens=500, reserve_output_tokens=100),
    ).build_request(
        static_segments=static,
        mission_segments=mission,
        working_segments=working,
        current_segments=current,
    )

    assert result.segments[0].name == "agent.rules"
    assert any(segment.metadata.get("trimmed") for segment in result.segments)


def test_conversation_memory_keeps_recent_window_and_summary():
    memory = ConversationMemory(task_id="node-1", max_recent_messages=2)
    memory.append("user", "one")
    memory.append("assistant", "two")
    memory.append("user", "three")

    assert memory.needs_compaction() is True
    memory.compact("summary of older turns")

    payload = memory.to_context_payload()
    assert payload["summary"] == "summary of older turns"
    assert [item["content"] for item in payload["recent_messages"]] == ["two", "three"]
