"""沙箱执行器：串行校验 → 执行 → 审计"""

import threading
from typing import Optional, Callable
from src.domain.sandbox.types import (
    ExecResult, ExecRequest, RiskLevel, ValidationResult
)
from src.domain.sandbox.validator import Validator


class Executor:
    """沙箱执行器统一接口（抽象基类）"""

    def exec(self, req: ExecRequest) -> ExecResult:
        raise NotImplementedError

    def backend(self) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError


class Sandbox:
    """封装 Validator + Executor + 审计回调"""

    def __init__(self, validator: Validator, executor: Executor):
        self._validator = validator
        self._executor = executor
        self._audit_fn: Optional[Callable[[ExecResult], None]] = None

    def set_audit_fn(self, fn: Callable[[ExecResult], None]):
        self._audit_fn = fn

    def get_backend(self) -> str:
        return self._executor.backend()

    def get_validator(self) -> Validator:
        return self._validator

    def exec(self, req: ExecRequest) -> ExecResult:
        # 1. 安全校验
        validation = self._validator.validate(req.command)

        result = ExecResult(
            command=req.command,
            validation=validation,
            backend=self._executor.backend(),
        )

        # 2. Block 级直接拒绝
        if validation.level == RiskLevel.BLOCK:
            result.exit_code = -1
            result.stderr = "[拒绝执行] " + validation.reason
            self._do_audit(result)
            return result

        # 3. Warn 级要求 confirm
        if validation.level == RiskLevel.WARN and not req.confirm:
            result.exit_code = -2
            violations = ", ".join(validation.violations)
            result.stderr = f"[需要确认] 该命令触发以下规则：{violations}；请重新调用并设置 confirm=true"
            self._do_audit(result)
            return result

        # 4. 进入沙箱执行
        exec_result = self._executor.exec(req)
        exec_result.command = req.command
        exec_result.validation = validation
        exec_result.backend = self._executor.backend()
        self._do_audit(exec_result)
        return exec_result

    def _do_audit(self, result: ExecResult):
        if self._audit_fn:
            t = threading.Thread(target=self._audit_fn, args=(result,), daemon=True)
            t.start()
