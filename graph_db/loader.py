from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_db.neo4j_client import Neo4jClient, GraphConfig

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DEFAULT_PARQUET = "data/user_features_1M_enriched.parquet"
DEFAULT_BATCH = 500


def _chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _run_batched(client, cypher, rows, batch_size, label):
    total, n_batches = (0, (len(rows) + batch_size - 1) // batch_size)
    for i, chunk in enumerate(_chunks(rows, batch_size), 1):
        client.run_query(cypher, {"rows": chunk}, write=True)
        total += len(chunk)
        if i % 20 == 0 or i == n_batches:
            logger.info("[%s] %d/%d batches — %d rows", label, i, n_batches, total)
    return total


UPSERT_ACCOUNTS = (
    "\nUNWIND $rows AS row\nMERGE (:Account {account_id: row.account_id})\n"
)


def load_accounts(client, df, batch_size):
    users = df["user_id"].drop_duplicates().tolist()
    logger.info("Loading %d Account nodes ...", len(users))
    rows = [{"account_id": str(u)} for u in users]
    return _run_batched(client, UPSERT_ACCOUNTS, rows, batch_size, "Account")


UPSERT_DEVICES = "\nUNWIND $rows AS row\nMERGE (d:Device {fingerprint: row.fingerprint})\nWITH d, row\nMATCH (a:Account {account_id: row.account_id})\nMERGE (a)-[:USED_DEVICE]->(d)\n"


def load_devices(client, df, batch_size):
    device_df = df[df["device_id"].notna()][["user_id", "device_id"]].drop_duplicates()
    logger.info("Loading %d AccountDevice edges ...", len(device_df))
    rows = [
        {"account_id": str(r.user_id), "fingerprint": str(r.device_id)}
        for r in device_df.itertuples()
    ]
    return _run_batched(client, UPSERT_DEVICES, rows, batch_size, "Device")


UPSERT_EMAIL_DOMAINS = "\nUNWIND $rows AS row\nMERGE (e:EmailDomain {domain: row.domain})\nWITH e, row\nMATCH (a:Account {account_id: row.account_id})\nMERGE (a)-[:HAS_EMAIL_DOMAIN]->(e)\n"


def load_email_domains(client, df, batch_size):
    email_df = df[df["email_domain"].notna()][
        ["user_id", "email_domain"]
    ].drop_duplicates()
    logger.info("Loading %d AccountEmailDomain edges ...", len(email_df))
    rows = [
        {"account_id": str(r.user_id), "domain": str(r.email_domain)}
        for r in email_df.itertuples()
    ]
    return _run_batched(client, UPSERT_EMAIL_DOMAINS, rows, batch_size, "EmailDomain")


UPSERT_IP_COUNTRIES = "\nUNWIND $rows AS row\nMERGE (ip:IPCountry {country: row.country})\nWITH ip, row\nMATCH (a:Account {account_id: row.account_id})\nMERGE (a)-[:REGISTERED_FROM]->(ip)\n"


def load_ip_countries(client, df, batch_size):
    ip_df = df[df["ip_country"].notna()][["user_id", "ip_country"]].drop_duplicates()
    logger.info("Loading %d AccountIPCountry edges ...", len(ip_df))
    rows = [
        {"account_id": str(r.user_id), "country": str(r.ip_country)}
        for r in ip_df.itertuples()
    ]
    return _run_batched(client, UPSERT_IP_COUNTRIES, rows, batch_size, "IPCountry")


UPSERT_PHONE_CARRIERS = "\nUNWIND $rows AS row\nMERGE (pc:PhoneCarrier {carrier_type: row.carrier_type})\nWITH pc, row\nMATCH (a:Account {account_id: row.account_id})\nMERGE (a)-[:USES_CARRIER]->(pc)\n"


def load_phone_carriers(client, df, batch_size):
    carrier_df = df[df["phone_carrier_type"].notna()][
        ["user_id", "phone_carrier_type"]
    ].drop_duplicates()
    logger.info("Loading %d AccountPhoneCarrier edges ...", len(carrier_df))
    rows = [
        {"account_id": str(r.user_id), "carrier_type": str(r.phone_carrier_type)}
        for r in carrier_df.itertuples()
    ]
    return _run_batched(client, UPSERT_PHONE_CARRIERS, rows, batch_size, "PhoneCarrier")


def load_all(parquet_path=DEFAULT_PARQUET, batch_size=DEFAULT_BATCH, schema_only=False):
    t0 = time.perf_counter()
    config = GraphConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    logger.info("=" * 55)
    logger.info("NEO4J GRAPH LOADER (connection graph only)")
    logger.info("URI : %s", config.uri)
    logger.info("Parquet: %s", parquet_path)
    logger.info("=" * 55)
    with Neo4jClient(config) as client:
        if not client.verify_connectivity():
            logger.error("Cannot reach Neo4j — is it running?")
            sys.exit(1)
        logger.info("Creating schema ...")
        client.create_schema()
        if schema_only:
            logger.info("--schema-only: done.")
            return
        logger.info("Loading parquet ...")
        df_full = pd.read_parquet(parquet_path)
        logger.info("Full: %d rows", len(df_full))
        train_idx, _ = train_test_split(
            df_full.index, test_size=0.2, random_state=42, stratify=df_full["is_fraud"]
        )
        df = df_full.loc[train_idx].reset_index(drop=True)
        logger.info(
            "Train (80%%): %d rows | Test excluded: %d rows",
            len(df),
            len(df_full) - len(df),
        )
        counts = {}
        logger.info("\nStep 1: Account nodes")
        counts["accounts"] = load_accounts(client, df, batch_size)
        logger.info("\nStep 2: Device edges (:Account)-[:USED_DEVICE]->(:Device)")
        counts["device_edges"] = load_devices(client, df, batch_size)
        logger.info(
            "\nStep 3: EmailDomain edges (:Account)-[:HAS_EMAIL_DOMAIN]->(:EmailDomain)"
        )
        counts["email_edges"] = load_email_domains(client, df, batch_size)
        logger.info(
            "\nStep 4: IPCountry edges (:Account)-[:REGISTERED_FROM]->(:IPCountry)"
        )
        counts["ip_country_edges"] = load_ip_countries(client, df, batch_size)
        logger.info(
            "\nStep 5: PhoneCarrier edges (:Account)-[:USES_CARRIER]->(:PhoneCarrier)"
        )
        counts["carrier_edges"] = load_phone_carriers(client, df, batch_size)
        elapsed = time.perf_counter() - t0
        logger.info("\n" + "=" * 55)
        logger.info("DONE in %.1fs", elapsed)
        for k, v in counts.items():
            logger.info("%-20s %d", k, v)
        logger.info("\nNode counts in Neo4j:")
        for label in ["Account", "Device", "EmailDomain", "IPCountry", "PhoneCarrier"]:
            r = client.run_query(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            logger.info("%-15s %d", label, r[0]["cnt"] if r else 0)
        logger.info("\nRelationship counts:")
        for rel in [
            "USED_DEVICE",
            "HAS_EMAIL_DOMAIN",
            "REGISTERED_FROM",
            "USES_CARRIER",
        ]:
            r = client.run_query(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt")
            logger.info("%-25s %d", rel, r[0]["cnt"] if r else 0)
        logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PARQUET)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    load_all(
        parquet_path=args.path, batch_size=args.batch, schema_only=args.schema_only
    )
