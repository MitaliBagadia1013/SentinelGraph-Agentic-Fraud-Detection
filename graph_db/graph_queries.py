from typing import Any, Dict, Tuple

WRITE_UPSERT_TRANSACTION = "\nMERGE (acc:Account {account_id: $account_id})\nON CREATE SET\n acc.created_at = $timestamp,\n acc.risk_score = 0.0,\n acc.total_transactions = 0,\n acc.fraud_transaction_count = 0\n\nMERGE (t:Transaction {transaction_id: $transaction_id})\nON CREATE SET\n t.amount = $amount,\n t.timestamp = $timestamp,\n t.fraud_score = $fraud_score,\n t.is_fraud = $is_fraud,\n t.merchant_id = $merchant_id\n\nMERGE (acc)-[:MADE]->(t)\n\nMERGE (m:Merchant {merchant_id: $merchant_id})\nON CREATE SET m.category = $merchant_category\nMERGE (t)-[:AT]->(m)\n\nSET acc.total_transactions =\n coalesce(acc.total_transactions, 0) + 1,\n acc.fraud_transaction_count =\n coalesce(acc.fraud_transaction_count, 0) +\n CASE WHEN $is_fraud = true THEN 1 ELSE 0 END\n\nRETURN acc.account_id AS account_id\n"
WRITE_LINK_DEVICE = "\nMERGE (d:Device {fingerprint: $fingerprint})\nWITH d\nMATCH (acc:Account {account_id: $account_id})\nMERGE (acc)-[:USES]->(d)\n"
WRITE_LINK_IP = "\nMERGE (ip:IPAddress {address: $ip_address})\nWITH ip\nMATCH (acc:Account {account_id: $account_id})\nMERGE (acc)-[:CONNECTED_FROM]->(ip)\n"
WRITE_LINK_P2P = "\nMERGE (sender:Account {account_id: $sender_id})\nMERGE (recipient:Account {account_id: $recipient_id})\nMERGE (sender)-[r:SENT_TO {transaction_id: $tx_id}]->(recipient)\nON CREATE SET r.amount = $amount, r.created_at = timestamp()\n"
WRITE_LINK_PHONE = "\nMERGE (p:PhoneNumber {number: $phone_number})\nWITH p\nMATCH (acc:Account {account_id: $account_id})\nMERGE (acc)-[:HAS_PHONE]->(p)\n"
WRITE_LINK_EMAIL = "\nMERGE (e:EmailAddress {address: $email_address})\nWITH e\nMATCH (acc:Account {account_id: $account_id})\nMERGE (acc)-[:HAS_EMAIL]->(e)\n"
WRITE_CREATE_FRAUD_RING = "\nMERGE (r:FraudRing {ring_id: $ring_id})\nON CREATE SET\n r.type = $ring_type,\n r.size = $size,\n r.confidence = $confidence,\n r.created_at = $created_at\nON MATCH SET\n r.size = $size,\n r.confidence = $confidence\n"
WRITE_ADD_RING_MEMBER = "\nMATCH (r:FraudRing {ring_id: $ring_id})\nMERGE (acc:Account {account_id: $account_id})\nMERGE (acc)-[:MEMBER_OF]->(r)\nSET acc.risk_score = 1.0\n"
WRITE_UPDATE_ACCOUNT_RISK = "\nMATCH (acc:Account {account_id: $account_id})\nSET acc.risk_score = $risk_score,\n acc.risk_updated_at = $updated_at\n"
MATCH_ACCOUNT_FRAUD_RING = "\nMATCH (acc:Account {account_id: $account_id})-[:MEMBER_OF]->(r:FraudRing)\nRETURN r.ring_id AS ring_id,\n r.size AS ring_size,\n r.type AS ring_type\nLIMIT 1\n"
MATCH_SHARED_DEVICE = "\nMATCH (d:Device {fingerprint: $fingerprint})<-[:USES]-(other:Account)\nWHERE other.account_id <> $account_id\nRETURN\n count(other) AS shared_count,\n sum(CASE WHEN other.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_accounts\n"
MATCH_SHARED_IP = "\nMATCH (ip:IPAddress {address: $ip_address})<-[:CONNECTED_FROM]-(other:Account)\nWHERE other.account_id <> $account_id\nRETURN\n count(other) AS shared_count,\n sum(CASE WHEN other.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_accounts\n"
MATCH_MULE_CHAIN = "\nMATCH path = (acc:Account {account_id: $account_id})\n -[:SENT_TO*1..5]->(end:Account)\nRETURN\n length(path) AS chain_depth,\n end.account_id AS endpoint_id,\n end.risk_score AS endpoint_risk\nORDER BY chain_depth DESC\nLIMIT 1\n"
MATCH_NETWORK_LINKS = "\nMATCH (acc:Account {account_id: $account_id})--(neighbor:Account)\nRETURN\n count(neighbor) AS total_connections,\n sum(CASE WHEN neighbor.fraud_transaction_count > 0 THEN 1 ELSE 0 END)\n AS fraud_links\n"
MATCH_ACCOUNT_HISTORY = "\nMATCH (acc:Account {account_id: $account_id})\nRETURN\n coalesce(acc.total_transactions, 0) AS total_tx,\n coalesce(acc.fraud_transaction_count, 0) AS fraud_tx,\n coalesce(acc.risk_score, 0) AS risk_score\n"
MATCH_ACCOUNT_VELOCITY = "\nMATCH (acc:Account {account_id: $account_id})-[:MADE]->(tx:Transaction)\nWHERE tx.timestamp >= $from_timestamp\nRETURN\n count(tx) AS tx_count,\n sum(tx.amount) AS total_amount\n"
MATCH_ALL_DEVICE_ACCOUNTS = "\nMATCH (d:Device {fingerprint: $fingerprint})<-[:USES]-(acc:Account)\nRETURN\n acc.account_id AS account_id,\n acc.risk_score AS risk_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n acc.total_transactions AS total_tx\nORDER BY acc.risk_score DESC\n"
MATCH_MONEY_MULE_CHAIN_FULL = "\nMATCH path = (start:Account {account_id: $account_id})\n -[:SENT_TO*1..5]->(end:Account)\nWITH path, nodes(path) AS hops, relationships(path) AS transfers\nUNWIND range(0, size(transfers)-1) AS i\nRETURN\n hops[i].account_id AS from_account,\n hops[i+1].account_id AS to_account,\n transfers[i].amount AS amount,\n i + 1 AS hop_number\nORDER BY hop_number\n"
MATCH_FRAUD_RING_MEMBERS = "\nMATCH (acc:Account {account_id: $account_id})-[:MEMBER_OF]->(r:FraudRing)\nMATCH (member:Account)-[:MEMBER_OF]->(r)\nRETURN\n member.account_id AS account_id,\n member.risk_score AS risk_score,\n member.fraud_transaction_count AS fraud_tx_count,\n r.ring_id AS ring_id,\n r.type AS ring_type\nORDER BY member.risk_score DESC\n"
MATCH_SHORTEST_PATH = "\nMATCH (source:Account {account_id: $source_id}),\n (target:Account {account_id: $target_id})\nMATCH path = shortestPath((source)-[*1..6]-(target))\nRETURN\n length(path) AS path_length,\n [n IN nodes(path) | coalesce(\n n.account_id, n.fingerprint, n.address, n.merchant_id, 'unknown'\n )] AS path_nodes,\n [r IN relationships(path) | type(r)] AS path_relationships\nLIMIT 1\n"
MATCH_HIGH_RISK_ACCOUNTS = "\nMATCH (acc:Account)\nWHERE acc.risk_score >= $min_risk_score\nRETURN\n acc.account_id AS account_id,\n acc.risk_score AS risk_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n acc.total_transactions AS total_tx\nORDER BY acc.risk_score DESC\nLIMIT $limit\n"
MATCH_ALL_FRAUD_RINGS = "\nMATCH (r:FraudRing)\nOPTIONAL MATCH (member:Account)-[:MEMBER_OF]->(r)\nRETURN\n r.ring_id AS ring_id,\n r.type AS ring_type,\n r.size AS ring_size,\n r.confidence AS confidence,\n r.created_at AS created_at,\n count(member) AS confirmed_member_count\nORDER BY r.created_at DESC\n"
MATCH_ACCOUNTS_SHARING_PHONE = "\nMATCH (p:PhoneNumber {number: $phone_number})<-[:HAS_PHONE]-(acc:Account)\nRETURN\n acc.account_id AS account_id,\n acc.risk_score AS risk_score\nORDER BY acc.risk_score DESC\n"
MATCH_ACCOUNTS_SHARING_EMAIL = "\nMATCH (e:EmailAddress {address: $email_address})<-[:HAS_EMAIL]-(acc:Account)\nRETURN\n acc.account_id AS account_id,\n acc.risk_score AS risk_score\nORDER BY acc.risk_score DESC\n"
ALGO_AGGREGATE_COMMUNITIES = "\nMATCH (acc:Account)\nWHERE acc.community_id IS NOT NULL\nWITH acc.community_id AS cid, collect(acc) AS members\nWHERE size(members) >= $min_size\nRETURN\n cid AS community_id,\n [m IN members | m.account_id] AS account_ids,\n size(members) AS size,\n avg(coalesce(m.risk_score, 0.0)) AS avg_risk,\n size([m IN members WHERE m.fraud_transaction_count > 0]) AS fraud_count\nORDER BY size DESC\n"
ALGO_TOP_PAGERANK_ACCOUNTS = "\nMATCH (acc:Account)\nWHERE acc.pagerank_score IS NOT NULL\nRETURN\n acc.account_id AS account_id,\n acc.pagerank_score AS pagerank_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n acc.risk_score AS risk_score\nORDER BY acc.pagerank_score DESC\nLIMIT $top_n\n"
ALGO_PAGERANK_FALLBACK = "\nMATCH (acc:Account)\nOPTIONAL MATCH (other:Account)-[:SENT_TO]->(acc)\nWITH acc, count(other) AS in_degree\nRETURN\n acc.account_id AS account_id,\n toFloat(in_degree) AS pagerank_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n acc.risk_score AS risk_score\nORDER BY in_degree DESC\nLIMIT $top_n\n"
ALGO_WCC_AGGREGATE = "\nMATCH (acc:Account)\nWHERE acc.wcc_component_id IS NOT NULL\nWITH acc.wcc_component_id AS comp_id, collect(acc) AS members\nWHERE size(members) >= $min_size\nRETURN\n comp_id AS component_id,\n size(members) AS size,\n size([m IN members WHERE m.fraud_transaction_count > 0]) AS fraud_count,\n avg(coalesce(m.risk_score, 0.0)) AS avg_risk\nORDER BY fraud_count DESC\n"
ALGO_LOUVAIN_FALLBACK = "\nMATCH (d:Device)<-[:USES]-(acc:Account)\nWITH d.fingerprint AS device_fp, collect(acc) AS members\nWHERE size(members) >= $min_size\nUNWIND members AS acc\nRETURN\n device_fp AS community_key,\n acc.account_id AS account_id,\n acc.risk_score AS risk_score,\n acc.fraud_transaction_count AS fraud_tx_count,\n size(members) AS community_size\nORDER BY community_size DESC\n"
ALGO_WCC_FALLBACK = "\nMATCH (ip:IPAddress)<-[:CONNECTED_FROM]-(acc:Account)\nWITH ip.address AS ip_addr, collect(acc) AS members\nWHERE size(members) >= $min_size\nRETURN\n ip_addr AS component_id,\n size(members) AS size,\n size([m IN members WHERE m.fraud_transaction_count > 0]) AS fraud_count,\n avg(coalesce(m.risk_score, 0.0)) AS avg_risk\nORDER BY fraud_count DESC\n"
ALGO_NODE_SIMILARITY = "\nMATCH (source:Account {account_id: $account_id})--(neighbor)\nWITH source, collect(id(neighbor)) AS source_neighbors\n\nMATCH (other:Account)--(other_neighbor)\nWHERE other.account_id <> $account_id\nWITH source, source_neighbors, other,\n collect(id(other_neighbor)) AS other_neighbors\n\nWITH source, other, source_neighbors, other_neighbors,\n [x IN source_neighbors WHERE x IN other_neighbors] AS intersection\nWITH source, other,\n size(intersection) AS shared_count,\n size(source_neighbors) + size(other_neighbors)\n - size(intersection) AS union_count\nWHERE union_count > 0\nWITH other, shared_count,\n toFloat(shared_count) / union_count AS similarity\nWHERE similarity >= $threshold\nRETURN\n other.account_id AS account_id,\n round(similarity * 1000) / 1000 AS similarity,\n shared_count AS shared_neighbors,\n other.risk_score AS risk_score,\n other.fraud_transaction_count AS fraud_tx_count\nORDER BY similarity DESC\nLIMIT $top_n\n"
DIAG_NODE_COUNTS = "\nMATCH (n)\nRETURN labels(n)[0] AS node_type, count(n) AS count\nORDER BY count DESC\n"
DIAG_RELATIONSHIP_COUNTS = "\nMATCH ()-[r]->()\nRETURN type(r) AS relationship_type, count(r) AS count\nORDER BY count DESC\n"
DIAG_FRAUD_RING_SUMMARY = "\nMATCH (r:FraudRing)\nRETURN\n count(r) AS total_rings,\n avg(r.size) AS avg_ring_size,\n max(r.size) AS largest_ring,\n sum(r.size) AS total_flagged_accounts\n"
DIAG_HIGH_RISK_SUMMARY = "\nMATCH (acc:Account)\nRETURN\n count(acc) AS total_accounts,\n count(CASE WHEN acc.risk_score >= 0.8 THEN 1 END) AS high_risk_count,\n count(CASE WHEN acc.risk_score >= 0.5\n AND acc.risk_score < 0.8 THEN 1 END) AS medium_risk_count,\n count(CASE WHEN acc.risk_score < 0.5 THEN 1 END) AS low_risk_count,\n avg(acc.risk_score) AS avg_risk_score\n"


class QueryLibrary:

    @staticmethod
    def account_fraud_ring(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_ACCOUNT_FRAUD_RING, {"account_id": account_id})

    @staticmethod
    def shared_device(fingerprint: str, account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (
            MATCH_SHARED_DEVICE,
            {"fingerprint": fingerprint, "account_id": account_id},
        )

    @staticmethod
    def shared_ip(ip_address: str, account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_SHARED_IP, {"ip_address": ip_address, "account_id": account_id})

    @staticmethod
    def mule_chain(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_MULE_CHAIN, {"account_id": account_id})

    @staticmethod
    def network_links(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_NETWORK_LINKS, {"account_id": account_id})

    @staticmethod
    def account_history(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_ACCOUNT_HISTORY, {"account_id": account_id})

    @staticmethod
    def account_velocity(
        account_id: str, from_timestamp: str
    ) -> Tuple[str, Dict[str, Any]]:
        return (
            MATCH_ACCOUNT_VELOCITY,
            {"account_id": account_id, "from_timestamp": from_timestamp},
        )

    @staticmethod
    def all_device_accounts(fingerprint: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_ALL_DEVICE_ACCOUNTS, {"fingerprint": fingerprint})

    @staticmethod
    def money_mule_chain_full(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_MONEY_MULE_CHAIN_FULL, {"account_id": account_id})

    @staticmethod
    def fraud_ring_members(account_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_FRAUD_RING_MEMBERS, {"account_id": account_id})

    @staticmethod
    def shortest_path(source_id: str, target_id: str) -> Tuple[str, Dict[str, Any]]:
        return (MATCH_SHORTEST_PATH, {"source_id": source_id, "target_id": target_id})

    @staticmethod
    def high_risk_accounts(
        min_risk_score: float = 0.7, limit: int = 100
    ) -> Tuple[str, Dict[str, Any]]:
        return (
            MATCH_HIGH_RISK_ACCOUNTS,
            {"min_risk_score": min_risk_score, "limit": limit},
        )

    @staticmethod
    def all_fraud_rings() -> Tuple[str, Dict[str, Any]]:
        return (MATCH_ALL_FRAUD_RINGS, {})

    @staticmethod
    def node_counts() -> Tuple[str, Dict[str, Any]]:
        return (DIAG_NODE_COUNTS, {})

    @staticmethod
    def relationship_counts() -> Tuple[str, Dict[str, Any]]:
        return (DIAG_RELATIONSHIP_COUNTS, {})

    @staticmethod
    def fraud_ring_summary() -> Tuple[str, Dict[str, Any]]:
        return (DIAG_FRAUD_RING_SUMMARY, {})

    @staticmethod
    def high_risk_summary() -> Tuple[str, Dict[str, Any]]:
        return (DIAG_HIGH_RISK_SUMMARY, {})
