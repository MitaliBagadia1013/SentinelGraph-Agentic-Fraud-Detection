from __future__ import annotations
import logging
import os
import time
from typing import Any, Dict, List, Optional
from feature_store.feature_schema import (
    FEATURE_DEFAULTS,
    FEATURE_NAMES,
    REDIS_FEATURE_NAMES,
    REDIS_FEATURE_DEFAULTS,
    apply_defaults,
)
from feature_store.redis_store import RedisFeatureStore
from feature_store.postgres_store import UserProfileStore

logger = logging.getLogger(__name__)
POSTGRES_TO_FEATURE_MAP: Dict[str, str] = {
    "total_transactions": "total_transactions_ever",
    "avg_monthly_spend": "avg_monthly_spend",
    "total_lifetime_value": "total_lifetime_value",
    "fraud_flag_count": "previous_fraud_incidents",
    "chargeback_count": "chargeback_count",
    "kyc_verified": "identity_verified",
    "is_high_risk": "is_high_risk_user",
    "account_type": "account_type",
    "country_of_residence": "country",
    "device_count": "device_count",
}


class FeatureLoader:

    def __init__(
        self,
        redis_store: Optional[RedisFeatureStore] = None,
        postgres_store: Optional[UserProfileStore] = None,
        skip_postgres: bool = False,
        skip_redis: bool = False,
    ) -> None:
        self._skip_postgres = skip_postgres
        self._skip_redis = skip_redis
        self._redis: Optional[RedisFeatureStore] = (
            None if skip_redis else redis_store or RedisFeatureStore()
        )
        self._postgres: Optional[UserProfileStore] = (
            None if skip_postgres else postgres_store or UserProfileStore()
        )
        logger.info(
            "FeatureLoader ready — redis=%s postgres=%s",
            not skip_redis,
            not skip_postgres,
        )

    def get_features(
        self, user_id: str, tx: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        features: Dict[str, Any] = dict(FEATURE_DEFAULTS)
        if not self._skip_postgres and self._postgres:
            try:
                profile = self._postgres.get_user_context(user_id)
                if profile:
                    for pg_col, feat_name in POSTGRES_TO_FEATURE_MAP.items():
                        if pg_col in profile and profile[pg_col] is not None:
                            features[feat_name] = profile[pg_col]
            except Exception as exc:
                logger.warning("Postgres lookup failed user=%s: %s", user_id, exc)
        if not self._skip_redis and self._redis:
            hot = self._redis.get(user_id)
            features.update(hot)
        if tx:
            features.update(tx)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("get_features user=%s latency=%.2fms", user_id, latency_ms)
        return features

    def get_features_batch(
        self, user_ids: List[str], tx_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        if not user_ids:
            return {}
        t0 = time.perf_counter()
        tx_map = tx_map or {}
        result: Dict[str, Dict[str, Any]] = {
            uid: dict(FEATURE_DEFAULTS) for uid in user_ids
        }
        if not self._skip_postgres and self._postgres:
            try:
                profiles = self._postgres.get_user_context_batch(user_ids)
                for uid, profile in profiles.items():
                    if profile:
                        for pg_col, feat_name in POSTGRES_TO_FEATURE_MAP.items():
                            if pg_col in profile and profile[pg_col] is not None:
                                result[uid][feat_name] = profile[pg_col]
            except Exception as exc:
                logger.warning("Postgres batch lookup failed: %s", exc)
        if not self._skip_redis and self._redis:
            hot_batch = self._redis.get_batch(user_ids)
            for uid, hot in hot_batch.items():
                result[uid].update(hot)
        for uid, tx in tx_map.items():
            if uid in result:
                result[uid].update(tx)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "get_features_batch users=%d latency=%.2fms", len(user_ids), latency_ms
        )
        return result

    def get_agent_context(self, user_id: str) -> Dict[str, Any]:
        if self._skip_postgres or not self._postgres:
            return {"user_id": user_id, "error": "postgres_disabled"}
        try:
            profile = self._postgres.get_user_context(user_id)
            if profile is None:
                return {"user_id": user_id, "error": "not_found"}
            return profile
        except Exception as exc:
            logger.error("get_agent_context user=%s: %s", user_id, exc)
            return {"user_id": user_id, "error": str(exc)}

    def update_velocity(self, user_id: str, **velocity_fields: Any) -> bool:
        if self._skip_redis or not self._redis:
            return False
        hot = {k: v for k, v in velocity_fields.items() if k in REDIS_FEATURE_NAMES}
        if not hot:
            logger.warning("update_velocity: no valid hot-feature keys provided")
            return False
        return self._redis.update_fields(user_id, hot)

    def ping(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        if self._redis:
            status["redis"] = self._redis.ping()
        else:
            status["redis"] = False
        if self._postgres:
            try:
                status["postgres"] = self._postgres.health_check()
            except Exception:
                status["postgres"] = False
        else:
            status["postgres"] = False
        return status

    def close(self) -> None:
        if self._redis:
            self._redis.close()

    def __enter__(self) -> "FeatureLoader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
