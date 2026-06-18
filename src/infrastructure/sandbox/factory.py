"""沙箱工厂：按 backend 字符串组装 Sandbox"""

import logging

from src.domain.sandbox.sandbox import Sandbox, Executor
from src.domain.sandbox.validator import Validator
from src.domain.sandbox.types import SandboxConfig, SecurityConfig
from src.infrastructure.sandbox.docker import DockerSandbox
from src.infrastructure.sandbox.local import LocalSandbox, MockSandbox

logger = logging.getLogger(__name__)


def new_sandbox(backend: str, sandbox_cfg: SandboxConfig,
                sec_cfg: SecurityConfig) -> Sandbox:
    """工厂函数：按 backend 字符串组装 Sandbox"""
    validator = Validator(sec_cfg)

    executor: Executor
    if backend == "docker":
        ds = DockerSandbox(sandbox_cfg)
        if ds.available():
            executor = ds
        else:
            logger.warning("Docker 不可用，沙箱降级到 mock 模式")
            executor = MockSandbox()
    elif backend == "local":
        executor = LocalSandbox(sandbox_cfg)
    elif backend == "mock":
        executor = MockSandbox()
    else:
        logger.warning(f"未知沙箱后端 {backend}，使用 mock")
        executor = MockSandbox()

    return Sandbox(validator, executor)
