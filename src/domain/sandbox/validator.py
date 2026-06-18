"""沙箱命令安全校验器：Block/Warn/Safe 三级"""

import re
from typing import List
from src.domain.sandbox.types import ValidationResult, RiskLevel, SecurityConfig
from dataclasses import dataclass


@dataclass
class Policy:
    level: RiskLevel
    pattern: str
    reason: str


# Block 级规则
_block_rules = [
    # 文件系统破坏
    (r"rm\s+(-[rfRF]+\s+)?/", "禁止删除根路径"),
    (r"rm\s+-[rfRF]*r[fF]*\s", "禁止 rm -rf"),
    (r"\bdd\s+if=", "禁止 dd 设备写入"),
    (r"\bmkfs\b", "禁止格式化文件系统"),
    (r">\s*/dev/(sd|hd|nvme|vd|xvd)", "禁止写入块设备"),
    (r":\s*\(\s*\)\s*\{.*:\s*\|", "禁止 Fork 炸弹"),
    # 权限提升
    (r"\bsudo\b", "禁止 sudo"),
    (r"\bsu\s", "禁止 su"),
    (r"\bchmod\s+[0-7]*7[0-7][0-7]\b", "禁止 chmod 777"),
    (r"\bchown\s+root\b", "禁止变更为 root 属主"),
    # 系统控制
    (r"\b(shutdown|reboot|halt|poweroff|init 0)\b", "禁止系统关机/重启"),
    (r"\bsystemctl\s+(stop|disable|mask)\b", "禁止停止系统服务"),
    (r"\biptables\b", "禁止修改防火墙规则"),
    # Shell 注入
    (r"\$\(", "禁止命令替换 $()"),
    ("`", "禁止反引号命令替换"),
    (r"\beval\b", "禁止 eval"),
    # 敏感文件
    (r"/etc/(passwd|shadow|sudoers|ssh)", "禁止访问系统凭证文件"),
    (r"~/?\.(ssh|aws|docker|kube)/", "禁止访问凭证目录"),
    # 路径遍历
    (r"\.\./\.\./", "禁止多级路径遍历"),
    # 网络
    (r"\b(curl|wget|nc|netcat|ncat)\s.*http", "禁止网络外联"),
    (r"\bssh\b", "禁止 SSH 连接"),
    # 进程滥用
    (r"\bkillall\b", "禁止 killall"),
    (r"\bnohup\b", "禁止 nohup 后台驻留"),
]

# Warn 级规则
_warn_rules = [
    (r"\brm\s", "删除文件操作"),
    (r">\s*\w", "输出重定向（可能覆盖文件）"),
    (r"\bkill\s", "进程终止操作"),
    (r"\bpip\s+install\b", "安装 Python 包"),
    (r"\bnpm\s+install\b", "安装 Node 包"),
    (r"\bapt(-get)?\s+install\b", "安装系统包"),
    (r"\bapk\s+add\b", "安装 Alpine 包"),
    (r";\s*\S", "命令链（分号分隔）"),
    (r"\|", "管道符"),
    (r"&&", "条件命令链 &&"),
    (r"\|\|", "条件命令链 ||"),
]

# 预编译正则
_block_patterns = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _block_rules]
_warn_patterns = [(re.compile(p, re.IGNORECASE), reason) for p, reason in _warn_rules]


class Validator:
    """对命令做静态安全校验"""

    def __init__(self, cfg: SecurityConfig):
        self._cfg = cfg

    def validate(self, command: str) -> ValidationResult:
        # 1. 长度检查
        if len(command) > self._cfg.max_command_length:
            return ValidationResult(level=RiskLevel.BLOCK, reason="命令超过最大长度限制")

        # 2. 空命令
        if not command.strip():
            return ValidationResult(level=RiskLevel.BLOCK, reason="命令不能为空")

        # 3. 白名单模式
        if self._cfg.allowlist_mode and self._cfg.allowlist:
            first_word = command.split()[0] if command.split() else ""
            if not any(first_word.lower() == a.lower() for a in self._cfg.allowlist):
                return ValidationResult(
                    level=RiskLevel.BLOCK,
                    reason=f'白名单模式：命令 "{first_word}" 未在允许列表中'
                )

        # 4. Block 规则
        for pattern, reason in _block_patterns:
            if pattern.search(command):
                return ValidationResult(level=RiskLevel.BLOCK, reason=reason)

        # 5. Warn 规则
        violations = []
        for pattern, reason in _warn_patterns:
            if pattern.search(command):
                violations.append(reason)

        if violations:
            return ValidationResult(level=RiskLevel.WARN, violations=violations)

        return ValidationResult(level=RiskLevel.SAFE)


def policy_snapshot() -> List[Policy]:
    """返回当前所有静态安全政策的只读快照"""
    out = []
    for (_, reason) in _block_rules:
        out.append(Policy(level=RiskLevel.BLOCK, pattern="", reason=reason))
    for (_, reason) in _warn_rules:
        out.append(Policy(level=RiskLevel.WARN, pattern="", reason=reason))
    return out
