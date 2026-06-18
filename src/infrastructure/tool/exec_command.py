"""exec_command 工具：通过 Sandbox 执行终端命令"""

import time as _time
from typing import Dict, Any

from src.domain.tool import Tool, Param
from src.domain.sandbox.sandbox import Sandbox
from src.domain.sandbox.types import ExecRequest, ExecResult, RiskLevel


def exec_command_tool(sb: Sandbox) -> Tool:
    """创建一个调用 Sandbox 执行终端命令的工具"""

    def _exec(params: Dict[str, Any]) -> str:
        cmd = params.get("command", "")
        if not isinstance(cmd, str) or not cmd.strip():
            return "错误：参数 command 不能为空"

        confirm = params.get("confirm", False)
        if isinstance(confirm, str):
            confirm = confirm.lower() in ("true", "1")

        result = sb.exec(ExecRequest(command=cmd, confirm=confirm))
        return _format_exec_result(result)

    return Tool(
        name="exec_command",
        description=(
            "在隔离沙箱中执行终端命令。支持 ls/cat/echo/python3/node 等常见操作；"
            "危险命令（rm -rf、sudo、网络外联等）会被自动拒绝；"
            "涉及删除/安装/管道等中等风险命令需通过 confirm=true 二次确认。"
        ),
        parameters=[
            Param(name="command", type="string", description="要执行的 Shell 命令", required=True),
            Param(name="confirm", type="boolean", description="对 warn 级命令的二次确认", required=False),
        ],
        execute=_exec,
    )


def _format_exec_result(r: ExecResult) -> str:
    parts = []

    # 安全级别提示
    if r.validation.level == RiskLevel.BLOCK:
        return f"🛑 **命令被拒绝**\n原因：{r.validation.reason}\n"
    if r.validation.level == RiskLevel.WARN:
        if r.exit_code == -2:
            violations = "、".join(r.validation.violations)
            return f"⚠️ **命令需要确认**\n触发规则：{violations}\n如确认无误，请设置 confirm=true 后重新执行。\n"
        parts.append(f"⚠️ 警告级命令已执行（触发规则：{'、'.join(r.validation.violations)}）\n")

    # 执行结果
    parts.append(f"**沙箱后端**: {r.backend} | **退出码**: {r.exit_code} | **耗时**: {r.duration:.2f}s\n")
    if r.killed:
        parts.append("⏱ 因超时被强制终止\n")
    if r.truncated:
        parts.append("✂️ 输出过长已被截断\n")

    if r.stdout:
        parts.append(f"\n**stdout**\n```\n{r.stdout}\n```\n")
    if r.stderr:
        parts.append(f"\n**stderr**\n```\n{r.stderr}\n```\n")
    if not r.stdout and not r.stderr:
        parts.append("（无输出）\n")

    return "".join(parts)
