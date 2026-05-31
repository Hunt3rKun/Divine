import sys
from pathlib import Path
import asyncio

import typer
from loguru import logger

app = typer.Typer(name="divine", help="多智能体自动化渗透测试框架")


@app.command()
def engage(
    config: Path = typer.Option(..., "--config", "-c", help="目标配置 YAML 文件"),
):
    """启动渗透测试"""
    from divine.config import DivineConfig
    from divine.session import Session

    cfg = DivineConfig.from_yaml(config)

    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        level=cfg.log_level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    )
    logger.add(
        "divine_debug.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
    )

    asyncio.run(Session(cfg).run())


@app.command()
def version():
    """显示版本号"""
    from divine import __version__
    typer.echo(f"Divine v{__version__}")
