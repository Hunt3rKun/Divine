from pathlib import Path
import asyncio

import typer

app = typer.Typer(name="divine", help="多智能体自动化渗透测试框架")


@app.command()
def engage(
    config: Path = typer.Option(..., "--config", "-c", help="目标配置 YAML 文件"),
):
    """启动渗透测试"""
    from divine.config import DivineConfig
    from divine.session import Session

    cfg = DivineConfig.from_yaml(config)
    asyncio.run(Session(cfg).run())


@app.command()
def version():
    """显示版本号"""
    from divine import __version__
    typer.echo(f"Divine v{__version__}")
