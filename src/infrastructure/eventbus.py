"""事件总线薄封装：连接 Kafka 时写消息，否则降级为日志"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Publisher:
    """事件发布者接口"""

    def publish(self, event_type: str, payload: str):
        raise NotImplementedError


class KafkaPublisher(Publisher):
    """Kafka 实现，不可用时降级为日志"""

    def __init__(self, producer=None, available: bool = False):
        self._producer = producer
        self._available = available and producer is not None

    def publish(self, event_type: str, payload: str):
        if self._available and self._producer:
            try:
                self._producer.produce(
                    self._producer_topic if hasattr(self, '_producer_topic') else "agi-events",
                    key=event_type.encode() if isinstance(event_type, str) else event_type,
                    value=payload.encode() if isinstance(payload, str) else payload,
                )
                self._producer.flush()
                return
            except Exception as e:
                logger.warning(f"Kafka 写入失败: {e}")
        logger.info(f"[Kafka-fallback] {event_type}: {payload}")


class LogPublisher(Publisher):
    """纯日志事件发布（Kafka 不可用时的默认实现）"""

    def publish(self, event_type: str, payload: str):
        logger.info(f"[Event] {event_type}: {payload}")
