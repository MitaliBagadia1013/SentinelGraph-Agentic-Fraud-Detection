from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "fraud_graph_2024"
    database: str = "neo4j"


SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
    "CREATE CONSTRAINT device_fp_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.fingerprint IS UNIQUE",
    "CREATE CONSTRAINT email_domain_unique IF NOT EXISTS FOR (e:EmailDomain) REQUIRE e.domain IS UNIQUE",
    "CREATE CONSTRAINT ip_country_unique IF NOT EXISTS FOR (i:IPCountry) REQUIRE i.country IS UNIQUE",
    "CREATE CONSTRAINT phone_carrier_unique IF NOT EXISTS FOR (p:PhoneCarrier) REQUIRE p.carrier_type IS UNIQUE",
]


class Neo4jClient:

    def __init__(self, config: GraphConfig):
        self.config = config
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._driver = GraphDatabase.driver(
                self.config.uri, auth=(self.config.username, self.config.password)
            )
            logger.debug("Neo4j driver created for %s", self.config.uri)
        except Exception as e:
            logger.error("Failed to create Neo4j driver: %s", e)
            self._driver = None

    def verify_connectivity(self) -> bool:
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning("Neo4j connectivity check failed: %s", e)
            return False

    def run_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None, write: bool = False
    ) -> List[Dict[str, Any]]:
        if self._driver is None:
            logger.warning("run_query called but driver is not initialised")
            return []
        params = params or {}
        try:
            with self._driver.session(database=self.config.database) as session:
                if write:
                    result = session.execute_write(
                        lambda tx: list(tx.run(cypher, **params))
                    )
                else:
                    result = session.execute_read(
                        lambda tx: list(tx.run(cypher, **params))
                    )
                return [dict(record) for record in result]
        except Exception as e:
            logger.error("Cypher query failed: %s | query: %.120s", e, cypher)
            return []

    def run_write(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        return self.run_query(cypher, params, write=True)

    def apply_schema(self) -> None:
        logger.info("Applying schema constraints...")
        for constraint in SCHEMA_CONSTRAINTS:
            try:
                self.run_write(constraint)
                logger.debug("Applied: %s", constraint[:60])
            except Exception as e:
                logger.warning("Schema constraint skipped (%s): %s", e, constraint[:60])
        logger.info("Schema constraints applied.")

    def health_check(self) -> Dict[str, Any]:
        if not self.verify_connectivity():
            return {"status": "unreachable", "uri": self.config.uri}
        try:
            node_count = self.run_query("MATCH (n) RETURN count(n) AS cnt")
            rel_count = self.run_query("MATCH ()-[r]->() RETURN count(r) AS cnt")
            return {
                "status": "ok",
                "uri": self.config.uri,
                "database": self.config.database,
                "node_count": node_count[0]["cnt"] if node_count else 0,
                "rel_count": rel_count[0]["cnt"] if rel_count else 0,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.debug("Neo4j driver closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return f"Neo4jClient(uri={self.config.uri!r}, db={self.config.database!r})"
