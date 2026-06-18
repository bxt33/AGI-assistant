"""Local/Mock 沙箱执行器"""

import logging
import subprocess
import time
from typing import Optional

from src.domain.sandbox.sandbox import Executor
from src.domain.sandbox.validator import Validator
from src.domain.sandbox.types import (
    ExecRequest, ExecResult, SandboxConfig, SecurityConfig, RiskLevel
)

logger = logging.getLogger(__name__)


class LocalSandbox(Executor):
    """在本机直接执行命令（无容器隔离），仅允许 Safe 级命令"""

    def __init__(self, cfg: SandboxConfig):
        self._cfg = cfg
        self._validator = Validator(SecurityConfig(
            max_command_length=cfg.max_output_bytes,
        ))

    def backend(self) -> str:
        return "local"

    def available(self) -> bool:
        return True

    def exec(self, req: ExecRequest) -> ExecResult:
        start = time.time()
        result = ExecResult(command=req.command, backend="local")

        timeout = req.timeout if req.timeout > 0 else self._cfg.timeout
        if timeout <= 0:
            timeout = 15.0

        # 本地模式只允许 Safe 级别命令
        v = self._validator.validate(req.command)
        if v.level != RiskLevel.SAFE:
            result.exit_code = -1
            result.stderr = f"[本地模式拒绝] 只允许 safe 级别命令，当前: {v.level.value} {v.violations}"
            return result

        try:
            proc = subprocess.run(
                ["sh", "-c", req.command],
                capture_output=True, text=True, timeout=timeout,
            )
            result.exit_code = proc.returncode
            max_bytes = self._cfg.max_output_bytes
            result.stdout = proc.stdout[:max_bytes] if max_bytes > 0 else proc.stdout
            result.stderr = proc.stderr[:max_bytes] if max_bytes > 0 else proc.stderr
        except subprocess.TimeoutExpired:
            result.killed = True
            result.exit_code = -4
            result.stderr = f"[超时] 执行超过 {timeout}s 被终止"

        result.duration = time.time() - start
        return result


class MockSandbox(Executor):
    """返回固定结果，用于测试或沙箱不可用时的占位"""

    def backend(self) -> str:
        return "mock"

    def available(self) -> bool:
        return True

    def exec(self, req: ExecRequest) -> ExecResult:
        return ExecResult(
            command=req.command,
            stdout=f"[mock] 命令 {req.command!r} 在模拟沙箱中执行（Docker 不可用）",
            exit_code=0,
            backend="mock",
            duration=0.001,
        )
