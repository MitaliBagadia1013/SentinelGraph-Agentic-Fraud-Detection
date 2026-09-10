from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import redis
from redis import Redis
from redis.connection import ConnectionPool
from redis.exceptions import RedisError
from sklearn.model_selection import train_test_split
from feature_store.feature_schema import (
    FEATURE_MAP,
    FEATURE_DEFAULTS,
    REDIS_FEATURE_NAMES,
    REDIS_FEATURE_DEFAULTS,
    apply_defaults,
    validate_feature_dict,
)

logger = logging.getLogger(__name__)
DEFAULT_TTL = int(os.getenv("REDIS_FEATURE_TTL", 86400))
DEFAULT_VELOCITY_TTL = int(os.getenv("REDIS_VELOCITY_TTL", 86400))
DEFAULT_BLACKLIST_TTL = int(os.getenv("REDIS_BLACKLIST_TTL", 604800))
KEY_PREFIX = "features"
VELOCITY_PREFIX = "velocity"
BLACKLIST_PREFIX = "blacklist"
WINDOW_1H = 3600
WINDOW_24H = 86400


def _user_key(user_id: str) -> str:
    return f"{KEY_PREFIX}:{user_id}"


def _velocity_tx_key(user_id: str) -> str:
    return f"{VELOCITY_PREFIX}:tx_count:{user_id}"


def _velocity_amt_key(user_id: str) -> str:
    return f"{VELOCITY_PREFIX}:tx_amount:{user_id}"


def _blacklist_key(entity_type: str) -> str:
    return f"{BLACKLIST_PREFIX}:{entity_type}"


def _cast_value(field: str, raw: str) -> Any:
    feature = FEATURE_MAP.get(field)
    if feature is None:
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    try:
        if feature.dtype is float:
            return float(raw)
        elif feature.dtype is int:
            return int(float(raw))
        elif feature.dtype is bool:
            return bool(int(float(raw)))
        else:
            return str(raw)
    except (ValueError, TypeError):
        return feature.default


def _encode_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


class RedisFeatureStore:

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", 6379)),
        db: int = int(os.getenv("REDIS_DB", 0)),
        password: Optional[str] = os.getenv("REDIS_PASSWORD"),
        max_connections: int = int(os.getenv("REDIS_MAX_CONNS", 50)),
        ttl: int = DEFAULT_TTL,
    ) -> None:
        self._ttl = ttl
        self._pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password if password else None,
            max_connections=max_connections,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )
        self._redis: Redis = Redis(connection_pool=self._pool)
        logger.info(
            "RedisFeatureStore initialised — %s:%s db=%s ttl=%ss", host, port, db, ttl
        )

    def ping(self) -> bool:
        try:
            return self._redis.ping()
        except RedisError as exc:
            logger.error("Redis ping failed: %s", exc)
            return False

    def get(self, user_id: str) -> Dict[str, Any]:
        key = _user_key(user_id)
        try:
            t0 = time.perf_counter()
            raw: Dict[str, str] = self._redis.hgetall(key)
            latency_ms = (time.perf_counter() - t0) * 1000
            if not raw:
                logger.debug("Cache MISS user=%s — returning defaults", user_id)
                return dict(REDIS_FEATURE_DEFAULTS)
            features = dict(REDIS_FEATURE_DEFAULTS)
            for field in REDIS_FEATURE_NAMES:
                if field in raw:
                    features[field] = _cast_value(field, raw[field])
            logger.debug("Cache HIT  user=%s  latency=%.2fms", user_id, latency_ms)
            return features
        except RedisError as exc:
            logger.error("Redis GET error user=%s: %s", user_id, exc)
            return dict(REDIS_FEATURE_DEFAULTS)

    def set(
        self,
        user_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
        validate: bool = False,
    ) -> bool:
        if validate:
            validate_feature_dict(features)
        hot = {
            k: features.get(k, REDIS_FEATURE_DEFAULTS.get(k, 0))
            for k in REDIS_FEATURE_NAMES
        }
        key = _user_key(user_id)
        mapping = {field: _encode_value(val) for field, val in hot.items()}
        effective_ttl = ttl if ttl is not None else self._ttl
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, effective_ttl)
            pipe.execute()
            logger.debug("Cache SET  user=%s  fields=%d", user_id, len(mapping))
            return True
        except RedisError as exc:
            logger.error("Redis SET error user=%s: %s", user_id, exc)
            return False

    def delete(self, user_id: str) -> bool:
        try:
            self._redis.delete(_user_key(user_id))
            return True
        except RedisError as exc:
            logger.error("Redis DELETE error user=%s: %s", user_id, exc)
            return False

    def ttl(self, user_id: str) -> int:
        try:
            return self._redis.ttl(_user_key(user_id))
        except RedisError:
            return -2

    def get_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not user_ids:
            return {}
        try:
            pipe = self._redis.pipeline(transaction=False)
            for uid in user_ids:
                pipe.hgetall(_user_key(uid))
            results = pipe.execute()
            output: Dict[str, Dict[str, Any]] = {}
            hits = misses = 0
            for uid, raw in zip(user_ids, results):
                if raw:
                    features = dict(REDIS_FEATURE_DEFAULTS)
                    for field in REDIS_FEATURE_NAMES:
                        if field in raw:
                            features[field] = _cast_value(field, raw[field])
                    output[uid] = features
                    hits += 1
                else:
                    output[uid] = dict(REDIS_FEATURE_DEFAULTS)
                    misses += 1
            logger.debug(
                "Cache batch GET  users=%d  hits=%d  misses=%d",
                len(user_ids),
                hits,
                misses,
            )
            return output
        except RedisError as exc:
            logger.error("Redis batch GET error: %s", exc)
            return {uid: dict(REDIS_FEATURE_DEFAULTS) for uid in user_ids}

    def set_batch(
        self,
        user_features: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None,
        validate: bool = False,
    ) -> int:
        if not user_features:
            return 0
        effective_ttl = ttl if ttl is not None else self._ttl
        try:
            pipe = self._redis.pipeline(transaction=False)
            for uid, features in user_features.items():
                if validate:
                    validate_feature_dict(features)
                hot = {
                    k: features.get(k, REDIS_FEATURE_DEFAULTS.get(k, 0))
                    for k in REDIS_FEATURE_NAMES
                }
                key = _user_key(uid)
                mapping = {f: _encode_value(v) for f, v in hot.items()}
                pipe.hset(key, mapping=mapping)
                pipe.expire(key, effective_ttl)
            pipe.execute()
            logger.info(
                "Cache batch SET  users=%d  ttl=%ss", len(user_features), effective_ttl
            )
            return len(user_features)
        except RedisError as exc:
            logger.error("Redis batch SET error: %s", exc)
            return 0

    def update_fields(
        self, user_id: str, fields: Dict[str, Any], reset_ttl: bool = True
    ) -> bool:
        if not fields:
            return True
        hot_fields = {k: v for k, v in fields.items() if k in REDIS_FEATURE_NAMES}
        if not hot_fields:
            logger.debug(
                "update_fields: no hot-feature keys found for user=%s", user_id
            )
            return True
        key = _user_key(user_id)
        mapping = {f: _encode_value(v) for f, v in hot_fields.items()}
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hset(key, mapping=mapping)
            if reset_ttl:
                pipe.expire(key, self._ttl)
            pipe.execute()
            return True
        except RedisError as exc:
            logger.error("Redis update_fields error user=%s: %s", user_id, exc)
            return False

    def warm_up_from_records(
        self,
        records: List[Dict[str, Any]],
        user_id_field: str = "user_id",
        chunk_size: int = 500,
    ) -> int:
        total = 0
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            batch: Dict[str, Dict[str, Any]] = {}
            for rec in chunk:
                uid = str(rec.get(user_id_field, ""))
                if not uid:
                    continue
                batch[uid] = {k: rec[k] for k in REDIS_FEATURE_NAMES if k in rec}
            total += self.set_batch(batch)
            logger.info("Warm-up progress: %d / %d users", total, len(records))
        return total

    def count_cached_users(self) -> int:
        count = 0
        cursor = 0
        pattern = f"{KEY_PREFIX}:*"
        try:
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
        except RedisError as exc:
            logger.error("Redis SCAN error: %s", exc)
        return count

    def close(self) -> None:
        self._pool.disconnect()
        logger.info("RedisFeatureStore connection pool closed")

    def __enter__(self) -> "RedisFeatureStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class RedisSlidingWindow:

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", 6379)),
        db: int = int(os.getenv("REDIS_DB", 0)),
        password: Optional[str] = os.getenv("REDIS_PASSWORD"),
        ttl: int = DEFAULT_VELOCITY_TTL,
    ) -> None:
        self._ttl = ttl
        pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password if password else None,
            max_connections=20,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )
        self._r: Redis = Redis(connection_pool=pool)
        logger.info("RedisSlidingWindow initialised — %s:%s ttl=%ss", host, port, ttl)

    def record_transaction(
        self, user_id: str, amount: float, timestamp: Optional[float] = None
    ) -> None:
        ts = timestamp if timestamp is not None else time.time()
        tx_key = _velocity_tx_key(user_id)
        amt_key = _velocity_amt_key(user_id)
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.zadd(tx_key, {f"{ts:.6f}": ts})
            pipe.zadd(amt_key, {f"{amount:.4f}|{ts:.6f}": ts})
            pipe.expire(tx_key, self._ttl)
            pipe.expire(amt_key, self._ttl)
            pipe.execute()
        except RedisError as e:
            logger.error("record_transaction failed user=%s: %s", user_id, e)

    def get_tx_count(self, user_id: str, window_s: int = WINDOW_1H) -> int:
        cutoff = time.time() - window_s
        key = _velocity_tx_key(user_id)
        try:
            self._r.zremrangebyscore(key, "-inf", cutoff)
            return self._r.zcard(key)
        except RedisError as e:
            logger.error("get_tx_count failed user=%s: %s", user_id, e)
            return 0

    def get_tx_amount(self, user_id: str, window_s: int = WINDOW_24H) -> float:
        cutoff = time.time() - window_s
        key = _velocity_amt_key(user_id)
        try:
            self._r.zremrangebyscore(key, "-inf", cutoff)
            members = self._r.zrange(key, 0, -1)
            total = 0.0
            for m in members:
                try:
                    total += float(m.split("|")[0])
                except (ValueError, IndexError):
                    pass
            return round(total, 2)
        except RedisError as e:
            logger.error("get_tx_amount failed user=%s: %s", user_id, e)
            return 0.0

    def get_velocity_summary(self, user_id: str) -> Dict[str, Any]:
        return {
            "tx_count_1h": self.get_tx_count(user_id, WINDOW_1H),
            "tx_count_24h": self.get_tx_count(user_id, WINDOW_24H),
            "tx_amount_1h": self.get_tx_amount(user_id, WINDOW_1H),
            "tx_amount_24h": self.get_tx_amount(user_id, WINDOW_24H),
        }

    def warm_user(
        self, user_id: str, timestamps: List[float], amounts: List[float]
    ) -> None:
        if not timestamps:
            return
        cutoff = time.time() - WINDOW_24H
        tx_key = _velocity_tx_key(user_id)
        amt_key = _velocity_amt_key(user_id)
        tx_mapping: Dict[str, float] = {}
        amt_mapping: Dict[str, float] = {}
        for ts, amt in zip(timestamps, amounts):
            if ts < cutoff:
                continue
            tx_mapping[f"{ts:.6f}"] = ts
            amt_mapping[f"{amt:.4f}|{ts:.6f}"] = ts
        if not tx_mapping:
            return
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.zadd(tx_key, tx_mapping)
            pipe.zadd(amt_key, amt_mapping)
            pipe.expire(tx_key, self._ttl)
            pipe.expire(amt_key, self._ttl)
            pipe.execute()
        except RedisError as e:
            logger.error("warm_user failed user=%s: %s", user_id, e)


class RedisBlacklistStore:

    def __init__(
        self,
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", 6379)),
        db: int = int(os.getenv("REDIS_DB", 0)),
        password: Optional[str] = os.getenv("REDIS_PASSWORD"),
        ttl: int = DEFAULT_BLACKLIST_TTL,
    ) -> None:
        self._ttl = ttl
        pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password if password else None,
            max_connections=10,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )
        self._r: Redis = Redis(connection_pool=pool)
        logger.info("RedisBlacklistStore initialised — %s:%s ttl=%ss", host, port, ttl)

    def add(self, entity_type: str, entities: List[str]) -> int:
        if not entities:
            return 0
        key = _blacklist_key(entity_type)
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.sadd(key, *entities)
            pipe.expire(key, self._ttl)
            results = pipe.execute()
            added = results[0]
            logger.info(
                "Blacklist[%s] +%d entities (total ~%d)",
                entity_type,
                added,
                self._r.scard(key),
            )
            return added
        except RedisError as e:
            logger.error("Blacklist add failed type=%s: %s", entity_type, e)
            return 0

    def remove(self, entity_type: str, entities: List[str]) -> int:
        if not entities:
            return 0
        key = _blacklist_key(entity_type)
        try:
            return self._r.srem(key, *entities)
        except RedisError as e:
            logger.error("Blacklist remove failed type=%s: %s", entity_type, e)
            return 0

    def is_blacklisted(self, entity_type: str, entity: str) -> bool:
        try:
            return bool(self._r.sismember(_blacklist_key(entity_type), entity))
        except RedisError as e:
            logger.error(
                "Blacklist check failed type=%s entity=%s: %s", entity_type, entity, e
            )
            return False

    def check_transaction(
        self, user_id: str, device_id: str, ip: str, email: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        reasons = []
        try:
            pipe = self._r.pipeline(transaction=False)
            pipe.sismember(_blacklist_key("user"), user_id)
            pipe.sismember(_blacklist_key("device"), device_id)
            pipe.sismember(_blacklist_key("ip"), ip)
            if email:
                domain = email.split("@")[-1].lower() if "@" in email else ""
                pipe.sismember(_blacklist_key("email_domain"), domain)
            results = pipe.execute()
        except RedisError as e:
            logger.error("Blacklist check_transaction failed: %s", e)
            return (False, [])
        if results[0]:
            reasons.append(f"BLACKLISTED_USER:{user_id}")
        if results[1]:
            reasons.append(f"BLACKLISTED_DEVICE:{device_id}")
        if results[2]:
            reasons.append(f"BLACKLISTED_IP:{ip}")
        if email and len(results) > 3 and results[3]:
            reasons.append(f"BLACKLISTED_EMAIL_DOMAIN:{email.split('@')[-1]}")
        return (bool(reasons), reasons)

    def size(self, entity_type: str) -> int:
        try:
            return self._r.scard(_blacklist_key(entity_type))
        except RedisError:
            return 0

    def bulk_add_from_parquet(
        self, path: str, lookback_hours: int = 24
    ) -> Dict[str, int]:
        logger.info(
            "Seeding blacklists from parquet (lookback=%dh) ...", lookback_hours
        )
        df_full = pd.read_parquet(path)
        train_idx, _ = train_test_split(
            df_full.index, test_size=0.2, random_state=42, stratify=df_full["is_fraud"]
        )
        df = df_full.loc[train_idx].reset_index(drop=True)
        logger.info(
            "  Using %d train rows (excluded %d test rows)",
            len(df),
            len(df_full) - len(df),
        )
        if "is_fraud" not in df.columns:
            logger.warning("'is_fraud' column not found — skipping blacklist seeding")
            return {}
        fraud_df = df[df["is_fraud"] == 1].copy()
        if "timestamp" in fraud_df.columns:
            try:
                fraud_df["timestamp"] = pd.to_datetime(fraud_df["timestamp"], utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
                fraud_df = fraud_df[fraud_df["timestamp"] >= cutoff]
            except Exception as e:
                logger.warning(
                    "Timestamp filtering failed: %s — using all fraud rows", e
                )
        counts: Dict[str, int] = {}
        if "user_id" in fraud_df.columns:
            uids = fraud_df["user_id"].dropna().astype(str).unique().tolist()
            counts["user"] = self.add("user", uids)
        if "device_id" in fraud_df.columns:
            devs = fraud_df["device_id"].dropna().astype(str).unique().tolist()
            counts["device"] = self.add("device", devs)
        for ip_col in ("ip_address", "ip"):
            if ip_col in fraud_df.columns:
                ips = fraud_df[ip_col].dropna().astype(str).unique().tolist()
                counts["ip"] = self.add("ip", ips)
                break
        logger.info(
            "Blacklist seeding complete: %s",
            " | ".join((f"{k}={v}" for k, v in counts.items())),
        )
        return counts


def warm_cache_from_parquet(
    parquet_path: str = "data/user_features_1M_enriched.parquet",
    lookback_hours: int = 24,
    top_n_users: int = 100000,
    feature_store: Optional[RedisFeatureStore] = None,
    sliding_window: Optional[RedisSlidingWindow] = None,
    blacklist_store: Optional[RedisBlacklistStore] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info(
        "CACHE WARMING STARTED (lookback=%dh, top_n=%d)", lookback_hours, top_n_users
    )
    logger.info("=" * 60)
    fs = feature_store or RedisFeatureStore()
    vw = sliding_window or RedisSlidingWindow()
    bl = blacklist_store or RedisBlacklistStore()
    logger.info("Step 0: Loading parquet ...")
    df_full = pd.read_parquet(parquet_path)
    logger.info("  Loaded %d rows, %d columns", len(df_full), len(df_full.columns))
    logger.info("  Applying train/test split (80%% train, 20%% held-out test)...")
    train_idx, _ = train_test_split(
        df_full.index, test_size=0.2, random_state=42, stratify=df_full["is_fraud"]
    )
    df = df_full.loc[train_idx].reset_index(drop=True)
    logger.info("  Train rows (warming Redis): %d", len(df))
    logger.info("  Test rows  (excluded — runtime data): %d", len(df_full) - len(df))
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
            recent_df = df[df["timestamp"] >= cutoff].copy()
            logger.info("  Rows in last %dh window: %d", lookback_hours, len(recent_df))
        except Exception as e:
            logger.warning("  Timestamp filter failed: %s — using full dataset", e)
            recent_df = df
    else:
        recent_df = df
    logger.info("Step 1: Warming feature hashes (top %d users) ...", top_n_users)
    from feature_store.feature_schema import FEATURE_NAMES

    if "transaction_count" in df.columns:
        top_users = (
            df.groupby("user_id")["transaction_count"]
            .max()
            .nlargest(top_n_users)
            .index.tolist()
        )
    else:
        top_users = df["user_id"].drop_duplicates().head(top_n_users).tolist()
    user_df = df[df["user_id"].isin(top_users)].drop_duplicates(
        subset=["user_id"], keep="last"
    )
    feature_batch: Dict[str, Dict[str, Any]] = {}
    for _, row in user_df.iterrows():
        uid = str(row["user_id"])
        features = {f: row[f] for f in FEATURE_NAMES if f in row.index}
        feature_batch[uid] = features
    features_warmed = fs.set_batch(feature_batch)
    logger.info("  Feature hashes warmed: %d users", features_warmed)
    logger.info("Step 2: Warming velocity sliding windows ...")
    velocity_warmed = 0
    if "timestamp" in recent_df.columns and "user_id" in recent_df.columns:
        recent_df["_epoch"] = recent_df["timestamp"].astype("int64") / 1000000000.0
        amt_col = "amount" if "amount" in recent_df.columns else None
        for uid, group in recent_df.groupby("user_id"):
            timestamps = group["_epoch"].tolist()
            amounts = group[amt_col].tolist() if amt_col else [0.0] * len(timestamps)
            vw.warm_user(str(uid), timestamps, amounts)
            velocity_warmed += 1
        logger.info("  Velocity windows warmed: %d users", velocity_warmed)
    else:
        logger.warning("  Skipping velocity warm-up: 'timestamp' column not found")
    logger.info(
        "Step 3: Seeding blacklists from confirmed fraud (last %dh) ...", lookback_hours
    )
    blacklist_counts = bl.bulk_add_from_parquet(parquet_path, lookback_hours)
    elapsed = time.perf_counter() - t0
    summary = {
        "features_warmed": features_warmed,
        "velocity_warmed": velocity_warmed,
        "blacklists": blacklist_counts,
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info("=" * 60)
    logger.info("CACHE WARMING COMPLETE in %.1fs", elapsed)
    logger.info("  Features cached : %d users", features_warmed)
    logger.info("  Velocity windows: %d users", velocity_warmed)
    logger.info(
        "  Blacklists seeded: %s",
        " | ".join((f"{k}={v}" for k, v in blacklist_counts.items())),
    )
    logger.info("=" * 60)
    return summary


if __name__ == "__main__":
    import sys, logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    path = (
        sys.argv[1] if len(sys.argv) > 1 else "data/user_features_1M_enriched.parquet"
    )
    warm_cache_from_parquet(path)
