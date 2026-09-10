import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from graph_db.neo4j_client import Neo4jClient, GraphConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphRiskResult:
    account_id: str
    is_known_fraud_ring_member: bool = False
    fraud_ring_id: Optional[str] = None
    fraud_ring_size: int = 0
    shared_device_account_count: int = 0
    shared_device_has_fraud_history: bool = False
    mule_chain_depth: int = 0
    mule_chain_endpoint_is_flagged: bool = False
    shared_ip_account_count: int = 0
    shared_ip_has_fraud_history: bool = False
    first_degree_fraud_links: int = 0
    total_connections: int = 0
    account_total_transactions: int = 0
    account_fraud_transaction_count: int = 0
    account_historical_fraud_rate: float = 0.0
    graph_risk_score: float = 0.0
    lookup_time_ms: float = 0.0
    graph_data_available: bool = True


@dataclass
class TransactionRecord:
    transaction_id: str
    account_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    timestamp: datetime
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    fraud_score: float = 0.0
    is_fraud: Optional[bool] = None
    recipient_account_id: Optional[str] = None


class FraudGraphDB:

    def __init__(self, config: GraphConfig):
        self.client = Neo4jClient(config)
        logger.info("FraudGraphDB initialized")

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "FraudGraphDB":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def record_transaction(self, tx: TransactionRecord) -> bool:
        cypher = "\n MERGE (acc:Account {account_id: $account_id})\n ON CREATE SET\n acc.created_at = $timestamp,\n acc.risk_score = 0.0,\n acc.total_transactions = 0,\n acc.fraud_transaction_count = 0\n\n MERGE (t:Transaction {transaction_id: $transaction_id})\n ON CREATE SET\n t.amount = $amount,\n t.timestamp = $timestamp,\n t.fraud_score = $fraud_score,\n t.is_fraud = $is_fraud,\n t.merchant_id = $merchant_id\n\n MERGE (acc)-[:MADE]->(t)\n\n MERGE (m:Merchant {merchant_id: $merchant_id})\n ON CREATE SET m.category = $merchant_category\n MERGE (t)-[:AT]->(m)\n\n SET acc.total_transactions = coalesce(acc.total_transactions, 0) + 1,\n acc.fraud_transaction_count =\n coalesce(acc.fraud_transaction_count, 0) +\n CASE WHEN $is_fraud = true THEN 1 ELSE 0 END\n\n RETURN acc.account_id AS account_id\n "
        result = self.client.run_query(
            cypher,
            {
                "account_id": tx.account_id,
                "transaction_id": tx.transaction_id,
                "amount": tx.amount,
                "timestamp": tx.timestamp.isoformat(),
                "fraud_score": tx.fraud_score,
                "is_fraud": tx.is_fraud if tx.is_fraud is not None else False,
                "merchant_id": tx.merchant_id,
                "merchant_category": tx.merchant_category,
            },
            write=True,
        )
        if not result:
            logger.error(f"Failed to write transaction {tx.transaction_id}")
            return False
        if tx.device_fingerprint:
            self._link_device(tx.account_id, tx.device_fingerprint)
        if tx.ip_address:
            self._link_ip(tx.account_id, tx.ip_address)
        if tx.recipient_account_id:
            self._link_p2p(
                tx.account_id, tx.recipient_account_id, tx.transaction_id, tx.amount
            )
        logger.debug(
            f"Recorded transaction {tx.transaction_id} for account {tx.account_id}"
        )
        return True

    def _link_device(self, account_id: str, fingerprint: str) -> None:
        self.client.run_query(
            "\n MERGE (d:Device {fingerprint: $fingerprint})\n WITH d\n MATCH (acc:Account {account_id: $account_id})\n MERGE (acc)-[:USES]->(d)\n ",
            {"account_id": account_id, "fingerprint": fingerprint},
            write=True,
        )

    def _link_ip(self, account_id: str, ip_address: str) -> None:
        self.client.run_query(
            "\n MERGE (ip:IPAddress {address: $ip_address})\n WITH ip\n MATCH (acc:Account {account_id: $account_id})\n MERGE (acc)-[:CONNECTED_FROM]->(ip)\n ",
            {"account_id": account_id, "ip_address": ip_address},
            write=True,
        )

    def _link_p2p(
        self, sender_id: str, recipient_id: str, transaction_id: str, amount: float
    ) -> None:
        self.client.run_query(
            "\n MERGE (sender:Account {account_id: $sender_id})\n MERGE (recipient:Account {account_id: $recipient_id})\n MERGE (sender)-[r:SENT_TO {transaction_id: $tx_id}]->(recipient)\n ON CREATE SET r.amount = $amount, r.created_at = timestamp()\n ",
            {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "tx_id": transaction_id,
                "amount": amount,
            },
            write=True,
        )

    def flag_fraud_ring(
        self,
        ring_id: str,
        account_ids: List[str],
        ring_type: str = "UNKNOWN",
        confidence: float = 1.0,
    ) -> bool:
        self.client.run_query(
            "\n MERGE (r:FraudRing {ring_id: $ring_id})\n ON CREATE SET r.type = $ring_type, r.size = $size,\n r.confidence = $confidence,\n r.created_at = $created_at\n ON MATCH SET r.size = $size, r.confidence = $confidence\n ",
            {
                "ring_id": ring_id,
                "ring_type": ring_type,
                "size": len(account_ids),
                "confidence": confidence,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            write=True,
        )
        for account_id in account_ids:
            self.client.run_query(
                "\n MATCH (r:FraudRing {ring_id: $ring_id})\n MERGE (acc:Account {account_id: $account_id})\n MERGE (acc)-[:MEMBER_OF]->(r)\n SET acc.risk_score = 1.0\n ",
                {"ring_id": ring_id, "account_id": account_id},
                write=True,
            )
        logger.info(
            f"Flagged fraud ring {ring_id} | type={ring_type} | members={len(account_ids)} | confidence={confidence:.2%}"
        )
        return True

    def update_account_risk(self, account_id: str, risk_score: float) -> None:
        self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})\n SET acc.risk_score = $risk_score,\n acc.risk_updated_at = $updated_at\n ",
            {
                "account_id": account_id,
                "risk_score": max(0.0, min(1.0, risk_score)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            write=True,
        )

    def get_account_graph_risk(
        self,
        account_id: str,
        device_fingerprint: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> GraphRiskResult:
        start = time.time()
        result = GraphRiskResult(account_id=account_id)
        try:
            self._enrich_fraud_ring_membership(result)
            if device_fingerprint:
                self._enrich_shared_device(result, device_fingerprint)
            if ip_address:
                self._enrich_shared_ip(result, ip_address)
            self._enrich_mule_chain(result)
            self._enrich_network_links(result)
            self._enrich_account_history(result)
            result.graph_risk_score = self._compute_graph_risk_score(result)
        except Exception as e:
            logger.error(f"Graph risk lookup failed for {account_id}: {e}")
            result.graph_data_available = False
            result.graph_risk_score = 0.0
        result.lookup_time_ms = (time.time() - start) * 1000
        logger.info(
            f"Graph risk | account={account_id} | score={result.graph_risk_score:.3f} | ring={result.is_known_fraud_ring_member} | shared_device={result.shared_device_account_count} | {result.lookup_time_ms:.0f}ms"
        )
        return result

    def _enrich_fraud_ring_membership(self, result: GraphRiskResult) -> None:
        rows = self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})-[:MEMBER_OF]->(r:FraudRing)\n RETURN r.ring_id AS ring_id, r.size AS ring_size\n LIMIT 1\n ",
            {"account_id": result.account_id},
        )
        if rows:
            result.is_known_fraud_ring_member = True
            result.fraud_ring_id = rows[0].get("ring_id")
            result.fraud_ring_size = rows[0].get("ring_size", 0)

    def _enrich_shared_device(
        self, result: GraphRiskResult, device_fingerprint: str
    ) -> None:
        rows = self.client.run_query(
            "\n MATCH (d:Device {fingerprint: $fingerprint})<-[:USES]-(other:Account)\n WHERE other.account_id <> $account_id\n RETURN\n count(other) AS shared_count,\n sum(CASE WHEN other.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_accounts\n ",
            {"fingerprint": device_fingerprint, "account_id": result.account_id},
        )
        if rows:
            result.shared_device_account_count = rows[0].get("shared_count", 0)
            result.shared_device_has_fraud_history = (
                rows[0].get("fraud_accounts", 0) > 0
            )

    def _enrich_shared_ip(self, result: GraphRiskResult, ip_address: str) -> None:
        rows = self.client.run_query(
            "\n MATCH (ip:IPAddress {address: $ip_address})<-[:CONNECTED_FROM]-(other:Account)\n WHERE other.account_id <> $account_id\n RETURN\n count(other) AS shared_count,\n sum(CASE WHEN other.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_accounts\n ",
            {"ip_address": ip_address, "account_id": result.account_id},
        )
        if rows:
            result.shared_ip_account_count = rows[0].get("shared_count", 0)
            result.shared_ip_has_fraud_history = rows[0].get("fraud_accounts", 0) > 0

    def _enrich_mule_chain(self, result: GraphRiskResult) -> None:
        rows = self.client.run_query(
            "\n MATCH path = (acc:Account {account_id: $account_id})\n -[:SENT_TO*1..5]->(end:Account)\n RETURN\n length(path) AS chain_depth,\n end.account_id AS endpoint_id,\n end.risk_score AS endpoint_risk\n ORDER BY chain_depth DESC\n LIMIT 1\n ",
            {"account_id": result.account_id},
        )
        if rows:
            result.mule_chain_depth = rows[0].get("chain_depth", 0)
            endpoint_risk = rows[0].get("endpoint_risk") or 0.0
            result.mule_chain_endpoint_is_flagged = float(endpoint_risk) > 0.7

    def _enrich_network_links(self, result: GraphRiskResult) -> None:
        rows = self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})--(neighbor:Account)\n RETURN\n count(neighbor) AS total_connections,\n sum(CASE WHEN neighbor.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_links\n ",
            {"account_id": result.account_id},
        )
        if rows:
            result.total_connections = rows[0].get("total_connections", 0)
            result.first_degree_fraud_links = rows[0].get("fraud_links", 0)

    def _enrich_account_history(self, result: GraphRiskResult) -> None:
        rows = self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})\n RETURN\n coalesce(acc.total_transactions, 0) AS total_tx,\n coalesce(acc.fraud_transaction_count, 0) AS fraud_tx\n ",
            {"account_id": result.account_id},
        )
        if rows:
            total = rows[0].get("total_tx", 0)
            fraud = rows[0].get("fraud_tx", 0)
            result.account_total_transactions = total
            result.account_fraud_transaction_count = fraud
            result.account_historical_fraud_rate = fraud / total if total > 0 else 0.0

    def _compute_graph_risk_score(self, result: GraphRiskResult) -> float:
        score = 0.0
        if result.is_known_fraud_ring_member:
            score += 0.4
        if result.shared_device_has_fraud_history:
            score += 0.25 * min(result.shared_device_account_count / 10.0, 1.0)
        if result.mule_chain_depth > 0:
            score += 0.2 * min(result.mule_chain_depth / 5.0, 1.0)
        if result.shared_ip_has_fraud_history:
            score += 0.1 * min(result.shared_ip_account_count / 10.0, 1.0)
        if result.first_degree_fraud_links > 0:
            score += 0.05 * min(result.first_degree_fraud_links / 5.0, 1.0)
        return round(min(score, 1.0), 4)

    def find_shared_device_accounts(
        self, device_fingerprint: str
    ) -> List[Dict[str, Any]]:
        return self.client.run_query(
            "\n MATCH (d:Device {fingerprint: $fingerprint})<-[:USES]-(acc:Account)\n RETURN\n acc.account_id AS account_id,\n acc.risk_score AS risk_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n acc.total_transactions AS total_tx\n ORDER BY acc.risk_score DESC\n ",
            {"fingerprint": device_fingerprint},
        )

    def find_money_mule_chain(
        self, account_id: str, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        return self.client.run_query(
            f"\n MATCH path = (start:Account {{account_id: $account_id}})\n -[:SENT_TO*1..{max_depth}]->(end:Account)\n WITH path, nodes(path) AS hops, relationships(path) AS transfers\n UNWIND range(0, size(transfers)-1) AS i\n RETURN\n hops[i].account_id AS from_account,\n hops[i+1].account_id AS to_account,\n transfers[i].amount AS amount,\n i + 1 AS hop_number\n ORDER BY hop_number\n ",
            {"account_id": account_id},
        )

    def get_fraud_ring_members(self, account_id: str) -> List[Dict[str, Any]]:
        return self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})-[:MEMBER_OF]->(r:FraudRing)\n MATCH (member:Account)-[:MEMBER_OF]->(r)\n RETURN\n member.account_id AS account_id,\n member.risk_score AS risk_score,\n member.fraud_transaction_count AS fraud_tx_count,\n r.ring_id AS ring_id,\n r.type AS ring_type\n ORDER BY member.risk_score DESC\n ",
            {"account_id": account_id},
        )

    def get_account_velocity(
        self, account_id: str, window_hours: int = 24
    ) -> Dict[str, Any]:
        from_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        rows = self.client.run_query(
            "\n MATCH (acc:Account {account_id: $account_id})-[:MADE]->(tx:Transaction)\n WHERE tx.timestamp >= $from_timestamp\n RETURN count(tx) AS tx_count, sum(tx.amount) AS total_amount\n ",
            {"account_id": account_id, "from_timestamp": from_timestamp},
        )
        if rows:
            return {
                "tx_count": rows[0].get("tx_count", 0),
                "total_amount": rows[0].get("total_amount", 0.0),
                "window_hours": window_hours,
            }
        return {"tx_count": 0, "total_amount": 0.0, "window_hours": window_hours}
