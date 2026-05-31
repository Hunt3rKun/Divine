import json
from pathlib import Path
from uuid import uuid4

import typer

app = typer.Typer(name="divine", help="多智能体自动化渗透测试框架")


@app.command()
def engage(
    config: Path = typer.Option(..., "--config", "-c", help="目标配置 YAML 文件"),
):
    """启动渗透测试编排"""
    from divine.blackboard import TaskContext
    from divine.config import DivineConfig
    from divine.llm import create_llm_client
    from divine.logger import LoggingSettings, configure_logging
    from divine.orchestrator import Orchestrator

    cfg = DivineConfig.from_yaml(config)

    logging_config = Path(cfg.logging_config)
    if logging_config.exists():
        configure_logging(config_path=logging_config)
    else:
        configure_logging(LoggingSettings())

    client = create_llm_client(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        config_path=cfg.llm_config,
    )
    context = TaskContext(
        task_id=cfg.task_id or f"task_{uuid4().hex[:8]}",
        goal=cfg.goal,
        target=cfg.target,
        scope=cfg.scope or [cfg.target],
        max_iterations=cfg.max_iterations,
        max_consecutive_failures=cfg.max_consecutive_failures,
    )

    result = Orchestrator(llm_client=client).run(context)
    summary = {
        "task_id": context.task_id,
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "node_count": len(result.blackboard.graph.nodes),
        "execution_count": len(result.blackboard.execution_results),
        "audit_count": len(result.blackboard.audit_feedback),
        "confirmed_fact_count": len(result.blackboard.intelligence.confirmed_facts),
        "last_event": result.blackboard.event_log[-1].event_type if result.blackboard.event_log else None,
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def version():
    """显示版本号"""
    from divine import __version__
    typer.echo(f"Divine v{__version__}")
