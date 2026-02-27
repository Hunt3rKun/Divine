
import logging
import sys
import os
from loguru import logger
from pathlib import Path

# 定义日志保存路径
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)  # 确保目录存在


def setup_logger():
    """
    配置 Loguru 日志记录器
    """
    # 1. 移除默认的处理器（Loguru 默认添加了一个输出到 stderr 的 handler）
    logger.remove()
    logger.configure(extra={"logger_name": "loguru"})
    # 2. 定义日志格式
    # 控制台格式：包含颜色、时间、级别、文件名、行号、消息
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[logger_name]}</cyan> "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    # 文件格式：不包含颜色代码，便于阅读和归档
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[logger_name]} "
        "{name}:{function}:{line} - "
        "{message}"
    )
    # 3. 添加控制台输出
    # level="INFO" 意味着 INFO 及以上级别会打印到屏幕
    logger.add(
        sys.stdout,
        format=log_format,
        level=os.getenv("LOG_LEVEL", "DEBUG"),  # 可通过环境变量控制
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    # 4. 添加通用文件日志 (所有级别)
    # rotation="100 MB": 文件超过 100MB 自动新建
    # retention="30 days": 日志保留 30 天
    # compression="zip": 自动压缩旧日志为 zip
    # enqueue=True: 异步写入，防止阻塞主线程 (对 Agent 循环非常重要)
    logger.add(
        LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="DEBUG",
        rotation="10 MB",
        retention="60 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8"
    )
    # 5. 添加错误日志文件 (只记录 ERROR 及以上)
    # 方便快速排查崩溃问题
    logger.add(
        LOG_DIR / "error_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="50 MB",
        retention="60 days",
        compression="zip",
        enqueue=True,
        encoding="utf-8"
    )
    # # 6. 拦截标准库 logging 的日志
    # # 这一步很重要，否则 litellm 等库的日志不会显示在 Loguru 中
    # class InterceptHandler(logging.Handler):
    #     def emit(self, record: logging.LogRecord):
    #         try:
    #             level = logger.level(record.levelname).name
    #         except Exception:
    #             level = record.levelno
    #
    #         frame = logging.currentframe()
    #         depth = 2
    #         this_file = os.path.abspath(__file__)
    #
    #         while frame:
    #             filename = os.path.abspath(frame.f_code.co_filename)
    #             if filename == logging.__file__ or filename == this_file:
    #                 frame = frame.f_back
    #                 depth += 1
    #             else:
    #                 break
    #
    #         logger.bind(logger_name=record.name).opt(
    #             depth=depth,
    #             exception=record.exc_info,
    #         ).log(level, record.getMessage())
    #
    # logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    return logger


# 初始化全局 logger 实例
log = setup_logger()
# 如果需要导出给其他模块使用
__all__ = ["log"]