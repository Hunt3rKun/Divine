from pathlib import Path
import json
import jinja2
from loguru import logger
from divine.blackboard.blackboard import Blackboard
from divine.llm.base import LLMMessage
from divine.llm.router import LLMRouter

REPORT_TEMPLATE_DIR = Path(__file__).parent / "templates"

class ReportGenerator:
    def __init__(self, router: LLMRouter, blackboard: Blackboard):
        self._router = router
        self._blackboard = blackboard
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(REPORT_TEMPLATE_DIR)),
            trim_blocks=True, lstrip_blocks=True,
        )

    async def generate(self, output_path: Path, model: str = "claude-sonnet-4-20250514") -> None:
        data = self._collect_data()
        narrative = await self._generate_narrative(data, model)
        html = self._render(data, narrative)
        output_path.write_text(html, encoding="utf-8")
        logger.info(f"报告已生成: {output_path}")

    def _collect_data(self) -> dict:
        return {
            "hosts": self._blackboard.read("hosts") or {},
            "ports": self._blackboard.read("ports") or {},
            "findings": self._blackboard.read("findings") or {},
            "credentials": self._blackboard.read("credentials") or {},
            "reflections": self._blackboard.read("reflections") or {},
        }

    async def _generate_narrative(self, data: dict, model: str) -> dict:
        prompt = f"""根据以下渗透测试数据，生成报告叙述部分。

数据:
{json.dumps(data, ensure_ascii=False, default=str)}

请返回 JSON 格式:
{{"executive_summary": "执行摘要", "attack_path": "攻击路径叙述", "risk_rating": "高/中/低", "recommendations": ["建议1", "建议2"]}}"""
        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = await self._router.chat(model, messages)
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"LLM 叙述生成失败: {e}")
            return {
                "executive_summary": "自动生成失败",
                "attack_path": "",
                "risk_rating": "未知",
                "recommendations": [],
            }

    def _render(self, data: dict, narrative: dict) -> str:
        template = self._env.get_template("report.jinja2")
        return template.render(data=data, narrative=narrative)
