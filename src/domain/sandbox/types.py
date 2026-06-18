"""沙箱执行类型定义"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class ValidationResult:
    level: RiskLevel = RiskLevel.SAFE
    violations: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ExecRequest:
    command: str = ""
    timeout: float = 0.0  # seconds
    confirm: bool = False


@dataclass
class ExecResult:
    command: str = ""
    validation: ValidationResult = field(default_factory=ValidationResult)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration: float = 0.0
    killed: bool = False
    backend: str = ""
    truncated: bool = False


@dataclass
class SandboxConfig:
    image: str = "ubuntu:22.04"
    timeout: float = 30.0
    max_output_bytes: int = 65536
    memory_limit_mb: int = 256
    cpu_percent: int = 50
    max_pids: int = 64
    network_disabled: bool = True
    read_only_rootfs: bool = True


@dataclass
class SecurityConfig:
    max_command_length: int = 500
    allowlist_mode: bool = False
    allowlist: List[str] = field(default_factory=list)
