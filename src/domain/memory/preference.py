"""用户偏好记忆：支持 LLM NER 提取 + 规则兜底"""

import threading
import re
from typing import Dict


class Preference:
    """用户偏好存储：内存 map + 持久化回调"""

    def __init__(self):
        self._mu = threading.RLock()
        self._data: Dict[str, str] = {}

    def save(self, key: str, value: str):
        with self._mu:
            self._data[key] = value

    def save_batch(self, kvs: Dict[str, str]):
        with self._mu:
            self._data.update(kvs)

    def load(self, user_id: str = "default") -> Dict[str, str]:
        with self._mu:
            return dict(self._data)

    def snapshot(self) -> Dict[str, str]:
        with self._mu:
            return dict(self._data)

    @staticmethod
    def extract_rule_based(msg: str) -> Dict[str, str]:
        """规则兜底：无 LLM 时从文本中提取偏好"""
        result = {}

        # "我喜欢xxx"
        m = re.search(r'我喜欢(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["喜好"] = m.group(1).strip()

        # "我叫xxx"
        m = re.search(r'我叫(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["姓名"] = m.group(1).strip()

        # "我在xxx"
        m = re.search(r'我在(.+?)(?:[，。！？\s]|$)', msg)
        if m:
            result["位置"] = m.group(1).strip()

        return result
