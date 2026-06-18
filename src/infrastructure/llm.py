"""LLM 客户端：OpenAI 兼容接口 + Mock 降级"""

import json
import logging
import re
from typing import List, Optional, Callable

import httpx

logger = logging.getLogger(__name__)


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class Client:
    """LLM 聊天客户端（OpenAI 兼容）"""

    def __init__(self, cfg):
        self._cfg = cfg
        self._http = httpx.Client(timeout=60.0)

    def chat(self, system_prompt: str, messages: List[Message]) -> str:
        if self._cfg.IsRealLLM:
            try:
                return self._call_api(system_prompt, messages)
            except Exception as e:
                logger.warning(f"LLM API 调用失败: {e}，回退到 Mock")
                return self._mock(messages)
        return self._mock(messages)

    def chat_stream(self, system_prompt: str, messages: List[Message],
                    on_token: Optional[Callable[[str], None]] = None) -> str:
        if not self._cfg.IsRealLLM:
            reply = self._mock(messages)
            if on_token:
                on_token(reply)
            return reply
        try:
            return self._call_api_stream(system_prompt, messages, on_token)
        except Exception as e:
            logger.warning(f"LLM 流式调用失败: {e}，回退到同步")
            return self.chat(system_prompt, messages)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self._cfg.EmbeddingAPIUrl or not self._cfg.EmbeddingAPIKey:
            return None
        try:
            # 多模态 embedding 端点
            api_url = self._cfg.EmbeddingAPIUrl
            if "/embeddings/multimodal" in api_url:
                inp = [{"type": "text", "text": text}]
            else:
                inp = text

            body = {
                "model": self._cfg.EmbeddingModel,
                "input": inp,
            }
            resp = self._http.post(
                api_url,
                json=body,
                headers={"Authorization": f"Bearer {self._cfg.EmbeddingAPIKey}"}
            )
            if resp.status_code != 200:
                logger.warning(f"Embedding API error: {resp.status_code}")
                return None
            data = resp.json()
            if "error" in data:
                logger.warning(f"Embedding API error: {data['error']}")
                return None

            if "/embeddings/multimodal" in api_url:
                embedding = data.get("data", {}).get("embedding", [])
            else:
                items = data.get("data", [])
                embedding = items[0].get("embedding", []) if items else []
            return embedding if embedding else None
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def extract_preferences(self, msg: str) -> dict:
        if not self._cfg.IsRealLLM:
            return _extract_rule_based(msg)
        try:
            prompt = (
                "从下面这句用户消息中，提取所有用户的个人信息和偏好，输出 JSON 对象"
                "（key为中文名称，value为具体值）。如果没有任何偏好信息，输出 {}。"
                "只输出 JSON，不要有其他内容。\n消息：" + msg
            )
            raw = self._call_api("", [Message(role="user", content=prompt)])
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
            return result if isinstance(result, dict) else {}
        except Exception:
            return _extract_rule_based(msg)

    def _call_api(self, system_prompt: str, messages: List[Message]) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            msgs.append(m.to_dict())

        body = {
            "model": self._cfg.LLMModel,
            "messages": msgs,
            "temperature": self._cfg.Temperature,
        }
        resp = self._http.post(
            self._cfg.LLMAPIUrl,
            json=body,
            headers={"Authorization": f"Bearer {self._cfg.LLMAPIKey}"}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"LLM API error: {data['error']}")
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LLM API returned empty choices")
        return choices[0]["message"]["content"]

    def _call_api_stream(self, system_prompt: str, messages: List[Message],
                         on_token: Optional[Callable[[str], None]] = None) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            msgs.append(m.to_dict())

        body = {
            "model": self._cfg.LLMModel,
            "messages": msgs,
            "temperature": self._cfg.Temperature,
            "stream": True,
        }
        full_reply = []

        with self._http.stream(
            "POST", self._cfg.LLMAPIUrl,
            json=body,
            headers={"Authorization": f"Bearer {self._cfg.LLMAPIKey}"}
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"LLM stream error {resp.status_code}")

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "error" in chunk:
                        raise RuntimeError(f"Stream error: {chunk['error']}")
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            full_reply.append(content)
                            if on_token:
                                on_token(content)
                except json.JSONDecodeError:
                    continue

        return "".join(full_reply)

    def _mock(self, messages: List[Message]) -> str:
        user_query = ""
        for m in messages:
            if m.role == "user":
                user_query = m.content
        q = user_query.lower()
        if "你是谁" in q:
            return "我是一个全能 AI 助手，具备知识库、工具调用、推理、记忆和稳定执行能力。"
        if "后端工程师" in q:
            return "后端工程师负责服务器端逻辑开发：API 设计、数据库、业务逻辑、系统架构、性能优化。常用 Go / Java / Python / MySQL / Redis。"
        return f"收到：「{user_query}」——这是模拟 LLM 回复，接入真实 API 后会更智能。"


def _extract_rule_based(msg: str) -> dict:
    result = {}
    m = re.search(r'我喜欢(.+?)(?:[，。！？\s]|$)', msg)
    if m:
        result["喜好"] = m.group(1).strip()
    m = re.search(r'我叫(.+?)(?:[，。！？\s]|$)', msg)
    if m:
        result["姓名"] = m.group(1).strip()
    return result
