"""Docker 沙箱：通过 docker CLI 在隔离容器中执行命令"""

import io
import logging
import subprocess
import time
from typing import Optional

from src.domain.sandbox.sandbox import Executor
from src.domain.sandbox.types import ExecRequest, ExecResult, SandboxConfig

logger = logging.getLogger(__name__)


class DockerSandbox(Executor):
    """通过 docker CLI 在容器内执行命令"""

    def __init__(self, cfg: SandboxConfig):
        self._cfg = cfg
        self._available = self._probe()

    def backend(self) -> str:
        return "docker"

    def available(self) -> bool:
        return self._available

    def exec(self, req: ExecRequest) -> ExecResult:
        start = time.time()
        result = ExecResult(command=req.command, backend="docker")

        if not self._available:
            result.exit_code = -3
            result.stderr = "Docker 后端不可用"
            return result

        timeout = req.timeout if req.timeout > 0 else self._cfg.timeout
        if timeout <= 0:
            timeout = 30.0

        args = self._build_docker_args(req.command)
        try:
            proc = subprocess.run(
                ["docker"] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            result.exit_code = proc.returncode
            result.stdout = self._limit_output(proc.stdout, self._cfg.max_output_bytes)
            result.stderr = self._limit_output(proc.stderr, self._cfg.max_output_bytes)

            # 截断标记
            if self._cfg.max_output_bytes > 0:
                max_bytes = self._cfg.max_output_bytes
                if len(proc.stdout) >= max_bytes or len(proc.stderr) >= max_bytes:
                    result.truncated = True

        except subprocess.TimeoutExpired:
            result.killed = True
            result.exit_code = -4
            result.stderr += f"\n[超时] 执行超过 {timeout}s 被强制终止"

        result.duration = time.time() - start
        return result

    def _build_docker_args(self, command: str) -> list:
        args = [
            "run", "--rm", "-i",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
        ]
        if self._cfg.network_disabled:
            args.extend(["--network", "none"])
        if self._cfg.read_only_rootfs:
            args.extend(["--read-only", "--tmpfs", "/tmp:rw,size=64m"])
        if self._cfg.memory_limit_mb > 0:
            args.extend(["--memory", f"{self._cfg.memory_limit_mb}m"])
        if self._cfg.cpu_percent > 0:
            args.extend(["--cpus", f"{self._cfg.cpu_percent / 100.0:.2f}"])
        if self._cfg.max_pids > 0:
            args.extend(["--pids-limit", str(self._cfg.max_pids)])

        image = self._cfg.image or "ubuntu:22.04"
        args.extend([image, "sh", "-c", command])
        return args

    def _probe(self) -> bool:
        try:
            subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=1.5
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _limit_output(text: str, max_bytes: int) -> str:
        if max_bytes <= 0:
            return text
        return text[:max_bytes]
