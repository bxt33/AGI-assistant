"""Kafka 连接薄封装"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from confluent_kafka import Producer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


def connect(cfg) -> tuple:
    """连接 Kafka，返回 (producer, status_string)"""
    if not HAS_KAFKA or not cfg.kafka_brokers:
        logger.warning("Kafka 未配置 broker (事件将输出到日志)")
        return None, "disconnected"

    try:
        producer = Producer({
            "bootstrap.servers": ",".join(cfg.kafka_brokers),
            "client.id": "agi-saber",
        })
        # 简单验证
        producer.list_topics(timeout=5)
        logger.info(f"✅ Kafka 已连接: {cfg.kafka_brokers}")
        return producer, "connected"
    except Exception as e:
        logger.warning(f"Kafka 连接失败: {e} (事件将输出到日志)")
        return None, "disconnected"
