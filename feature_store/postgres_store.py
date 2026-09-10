import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from psycopg2.pool import ThreadedConnectionPool
from sklearn.model_selection import train_test_split

load_dotenv()
logger = logging.getLogger(__name__)
POSTGRES_DSN = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:Mitali%40123@localhost:5432/fraud_detection"
)
POOL_MIN = 2
POOL_MAX = 10
TABLE_NAME = "user_profiles"
PARQUET_PATH = "data/user_features_1M_enriched.parquet"
TRAIN_TEST_SPLIT_SIZE = 0.2
TRAIN_TEST_RANDOM_STATE = 42
PROFILE_COLUMNS = [
    "user_id",
    "email",
    "phone",
    "account_status",
    "kyc_verified",
    "account_type",
    "country_of_residence",
    "account_created_at",
    "primary_device_id",
    "device_count",
    "fraud_flag_count",
    "chargeback_count",
    "is_high_risk",
    "total_lifetime_value",
    "avg_monthly_spend",
    "total_transactions",
    "last_login_at",
    "last_transaction_at",
]
CREATE_TABLE_SQL = f"\nCREATE TABLE IF NOT EXISTS {TABLE_NAME} (\n user_id VARCHAR(64) PRIMARY KEY,\n email VARCHAR(255),\n phone VARCHAR(32),\n account_status VARCHAR(20) NOT NULL DEFAULT 'Active',\n kyc_verified BOOLEAN NOT NULL DEFAULT FALSE,\n account_type VARCHAR(20) NOT NULL DEFAULT 'personal',\n country_of_residence VARCHAR(8),\n account_created_at TIMESTAMPTZ,\n primary_device_id VARCHAR(64),\n device_count INTEGER NOT NULL DEFAULT 1,\n fraud_flag_count INTEGER NOT NULL DEFAULT 0,\n chargeback_count INTEGER NOT NULL DEFAULT 0,\n is_high_risk BOOLEAN NOT NULL DEFAULT FALSE,\n total_lifetime_value DECIMAL(14, 2) NOT NULL DEFAULT 0.0,\n avg_monthly_spend DECIMAL(12, 2) NOT NULL DEFAULT 0.0,\n total_transactions INTEGER NOT NULL DEFAULT 0,\n last_login_at TIMESTAMPTZ,\n last_transaction_at TIMESTAMPTZ,\n updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n);\n\nCREATE INDEX IF NOT EXISTS idx_user_profiles_status\n ON {TABLE_NAME} (account_status);\n\nCREATE INDEX IF NOT EXISTS idx_user_profiles_high_risk\n ON {TABLE_NAME} (is_high_risk) WHERE is_high_risk = TRUE;\n\nCREATE INDEX IF NOT EXISTS idx_user_profiles_last_tx\n ON {TABLE_NAME} (last_transaction_at DESC);\n"
_pool: Optional[ThreadedConnectionPool] = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(POOL_MIN, POOL_MAX, POSTGRES_DSN)
        logger.info(
            "PostgreSQL connection pool created (min=%d, max=%d)", POOL_MIN, POOL_MAX
        )
    return _pool


def _get_conn():
    return _get_pool().getconn()


def _put_conn(conn):
    _get_pool().putconn(conn)


class UserProfileStore:

    def __init__(self):
        logger.info("UserProfileStore initialised (table=%s)", TABLE_NAME)

    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        sql = f"SELECT * FROM {TABLE_NAME} WHERE user_id = %s LIMIT 1"
        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (str(user_id),))
                row = cur.fetchone()
            if row is None:
                return None
            return _serialize_profile(dict(row))
        except Exception as e:
            logger.error("get_user_context failed for %s: %s", user_id, e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def get_user_context_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not user_ids:
            return {}
        sql = f"SELECT * FROM {TABLE_NAME} WHERE user_id = ANY(%s)"
        conn = _get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (list(user_ids),))
                rows = cur.fetchall()
            return {str(row["user_id"]): _serialize_profile(dict(row)) for row in rows}
        except Exception as e:
            logger.error("get_user_context_batch failed: %s", e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def upsert_profile(
        self,
        user_id: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        account_status: str = "Active",
        kyc_verified: bool = False,
        account_type: str = "personal",
        country_of_residence: Optional[str] = None,
        account_created_at: Optional[datetime] = None,
        primary_device_id: Optional[str] = None,
        device_count: int = 1,
        fraud_flag_count: int = 0,
        chargeback_count: int = 0,
        total_lifetime_value: float = 0.0,
        avg_monthly_spend: float = 0.0,
        total_transactions: int = 0,
        last_login_at: Optional[datetime] = None,
        last_transaction_at: Optional[datetime] = None,
    ) -> None:
        is_high_risk = fraud_flag_count > 0 or chargeback_count >= 2
        sql = f"\n INSERT INTO {TABLE_NAME} (\n user_id, email, phone, account_status, kyc_verified,\n account_type, country_of_residence, account_created_at,\n primary_device_id, device_count, fraud_flag_count,\n chargeback_count, is_high_risk, total_lifetime_value,\n avg_monthly_spend, total_transactions,\n last_login_at, last_transaction_at, updated_at\n ) VALUES (\n %s, %s, %s, %s, %s, %s, %s, %s,\n %s, %s, %s, %s, %s, %s, %s, %s,\n %s, %s, NOW()\n )\n ON CONFLICT (user_id) DO UPDATE SET\n email = COALESCE(EXCLUDED.email, {TABLE_NAME}.email),\n phone = COALESCE(EXCLUDED.phone, {TABLE_NAME}.phone),\n account_status = EXCLUDED.account_status,\n kyc_verified = EXCLUDED.kyc_verified,\n account_type = EXCLUDED.account_type,\n country_of_residence = COALESCE(EXCLUDED.country_of_residence, {TABLE_NAME}.country_of_residence),\n account_created_at = COALESCE(EXCLUDED.account_created_at, {TABLE_NAME}.account_created_at),\n primary_device_id = COALESCE(EXCLUDED.primary_device_id, {TABLE_NAME}.primary_device_id),\n device_count = GREATEST(EXCLUDED.device_count, {TABLE_NAME}.device_count),\n fraud_flag_count = EXCLUDED.fraud_flag_count,\n chargeback_count = EXCLUDED.chargeback_count,\n is_high_risk = EXCLUDED.is_high_risk,\n total_lifetime_value = EXCLUDED.total_lifetime_value,\n avg_monthly_spend = EXCLUDED.avg_monthly_spend,\n total_transactions = EXCLUDED.total_transactions,\n last_login_at = GREATEST(EXCLUDED.last_login_at, {TABLE_NAME}.last_login_at),\n last_transaction_at = GREATEST(EXCLUDED.last_transaction_at, {TABLE_NAME}.last_transaction_at),\n updated_at = NOW()\n "
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        str(user_id),
                        email,
                        phone,
                        account_status,
                        kyc_verified,
                        account_type,
                        country_of_residence,
                        account_created_at,
                        primary_device_id,
                        int(device_count),
                        int(fraud_flag_count),
                        int(chargeback_count),
                        is_high_risk,
                        round(float(total_lifetime_value), 2),
                        round(float(avg_monthly_spend), 2),
                        int(total_transactions),
                        last_login_at,
                        last_transaction_at,
                    ),
                )
            conn.commit()
        except Exception as e:
            logger.error("upsert_profile failed for %s: %s", user_id, e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def flag_user(self, user_id: str, reason: str = "") -> None:
        sql = f"\n UPDATE {TABLE_NAME}\n SET fraud_flag_count = fraud_flag_count + 1,\n is_high_risk = TRUE,\n account_status = 'Flagged',\n updated_at = NOW()\n WHERE user_id = %s\n "
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (str(user_id),))
            conn.commit()
            logger.info("User %s flagged. Reason: %s", user_id, reason)
        except Exception as e:
            logger.error("flag_user failed for %s: %s", user_id, e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def freeze_account(self, user_id: str) -> None:
        sql = f"\n UPDATE {TABLE_NAME}\n SET account_status = 'Frozen',\n is_high_risk = TRUE,\n updated_at = NOW()\n WHERE user_id = %s\n "
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (str(user_id),))
            conn.commit()
            logger.info("Account frozen for user %s", user_id)
        except Exception as e:
            logger.error("freeze_account failed for %s: %s", user_id, e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def bulk_load_from_parquet(
        self, path: str = PARQUET_PATH, chunk_size: int = 10000, train_only: bool = True
    ) -> int:
        logger.info("Bulk loading user profiles from %s ...", path)
        df = pd.read_parquet(path)
        logger.info("Read %d rows, %d columns", len(df), len(df.columns))
        if train_only:
            if "is_fraud" not in df.columns:
                raise ValueError("Column 'is_fraud' required for stratified split")
            train_df, test_df = train_test_split(
                df,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=TRAIN_TEST_RANDOM_STATE,
                stratify=df["is_fraud"],
            )
            logger.info(
                "Train split: %d rows | Test split: %d rows (stays in parquet only)",
                len(train_df),
                len(test_df),
            )
            df = train_df
        profiles = _extract_profiles(df)
        logger.info("Extracted %d identity profiles", len(profiles))
        total = 0
        n_chunks = (len(profiles) + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            chunk = profiles[i * chunk_size : (i + 1) * chunk_size]
            written = self._upsert_profile_batch(chunk)
            total += written
            logger.info(
                "chunk %d/%d — %d profiles written (total %d)",
                i + 1,
                n_chunks,
                written,
                total,
            )
        logger.info("Bulk load complete: %d user profiles written to PostgreSQL", total)
        return total

    def initialise_table(self) -> None:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()
            logger.info("Table '%s' initialised", TABLE_NAME)
        except Exception as e:
            logger.error("initialise_table failed: %s", e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)

    def health_check(self) -> bool:
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            _put_conn(conn)
            return True
        except Exception as e:
            logger.warning("PostgreSQL health check failed: %s", e)
            return False

    def _upsert_profile_batch(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        sql = f"\n INSERT INTO {TABLE_NAME} (\n user_id, email, phone, account_status, kyc_verified,\n account_type, country_of_residence, account_created_at,\n primary_device_id, device_count, fraud_flag_count,\n chargeback_count, is_high_risk, total_lifetime_value,\n avg_monthly_spend, total_transactions,\n last_login_at, last_transaction_at, updated_at\n ) VALUES (\n %(user_id)s, %(email)s, %(phone)s, %(account_status)s,\n %(kyc_verified)s, %(account_type)s, %(country_of_residence)s,\n %(account_created_at)s, %(primary_device_id)s, %(device_count)s,\n %(fraud_flag_count)s, %(chargeback_count)s, %(is_high_risk)s,\n %(total_lifetime_value)s, %(avg_monthly_spend)s,\n %(total_transactions)s, %(last_login_at)s,\n %(last_transaction_at)s, NOW()\n )\n ON CONFLICT (user_id) DO UPDATE SET\n device_count = GREATEST(EXCLUDED.device_count, {TABLE_NAME}.device_count),\n fraud_flag_count = EXCLUDED.fraud_flag_count,\n chargeback_count = EXCLUDED.chargeback_count,\n is_high_risk = EXCLUDED.is_high_risk,\n total_lifetime_value = EXCLUDED.total_lifetime_value,\n avg_monthly_spend = EXCLUDED.avg_monthly_spend,\n total_transactions = EXCLUDED.total_transactions,\n last_transaction_at = GREATEST(EXCLUDED.last_transaction_at, {TABLE_NAME}.last_transaction_at),\n updated_at = NOW()\n "
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
            conn.commit()
            return len(rows)
        except Exception as e:
            logger.error("_upsert_profile_batch failed: %s", e)
            conn.rollback()
            raise
        finally:
            _put_conn(conn)


def _extract_profiles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profiles = []
    for _, row in df.iterrows():
        uid = str(row.get("user_id", ""))
        if not uid:
            continue
        avg_amt = _safe_float(row.get("avg_transaction_amount", 0))
        txn_count = _safe_int(row.get("transaction_count", 0))
        total_ltv = round(avg_amt * txn_count, 2)
        avg_monthly = round(avg_amt * (txn_count / 24), 2) if txn_count else 0.0
        chargeback_count = _safe_int(row.get("chargeback_count", 0))
        fraud_flag_count = _safe_int(row.get("declined_transaction_count", 0))
        is_high_risk = bool(fraud_flag_count > 2 or chargeback_count >= 2)
        profiles.append(
            {
                "user_id": uid,
                "email": None,
                "phone": None,
                "account_status": "Active",
                "kyc_verified": bool(_safe_int(row.get("kyc_verified", 0))),
                "account_type": str(row.get("account_type", "personal") or "personal"),
                "country_of_residence": str(row.get("country_code", "") or "")[:8]
                or None,
                "account_created_at": None,
                "primary_device_id": str(row.get("device_id", "") or "")[:64] or None,
                "device_count": max(1, _safe_int(row.get("device_count", 1))),
                "fraud_flag_count": fraud_flag_count,
                "chargeback_count": chargeback_count,
                "is_high_risk": is_high_risk,
                "total_lifetime_value": total_ltv,
                "avg_monthly_spend": avg_monthly,
                "total_transactions": txn_count,
                "last_login_at": None,
                "last_transaction_at": _safe_ts(row.get("timestamp")),
            }
        )
    return profiles


def _serialize_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for k, v in row.items():
        if k == "updated_at":
            continue
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        elif hasattr(v, "item"):
            result[k] = v.item()
        else:
            result[k] = v
    return result


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _safe_ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, datetime):
        return v
    try:
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


PostgresFeatureStore = UserProfileStore
