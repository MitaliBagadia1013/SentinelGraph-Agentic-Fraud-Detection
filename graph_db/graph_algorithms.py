import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from graph_db.neo4j_client import Neo4jClient, GraphConfig

logger = logging.getLogger(__name__)


@dataclass
class CommunityResult:
    community_id: int
    account_ids: List[str] = field(default_factory=list)
    size: int = 0
    fraud_account_count: int = 0
    fraud_rate: float = 0.0
    avg_risk_score: float = 0.0
    is_suspicious: bool = False
    suspicion_reason: str = ""


@dataclass
class PageRankResult:
    account_id: str
    pagerank_score: float = 0.0
    fraud_transaction_count: int = 0
    risk_score: float = 0.0
    is_hub: bool = False


@dataclass
class ShortestPathResult:
    source_account: str
    target_account: str
    path_length: int = 0
    path_nodes: List[str] = field(default_factory=list)
    path_relationships: List[str] = field(default_factory=list)
    connected: bool = False


class FraudGraphAlgorithms:
    SUSPICIOUS_FRAUD_RATE_THRESHOLD = 0.3
    SUSPICIOUS_MIN_SIZE = 3
    PAGERANK_HUB_THRESHOLD = 1.0

    def __init__(self, config: GraphConfig):
        self.client = Neo4jClient(config)
        self._gds_available = self._check_gds()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "FraudGraphAlgorithms":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _check_gds(self) -> bool:
        try:
            result = self.client.run_query("RETURN gds.version() AS version")
            version = result[0].get("version") if result else None
            if version:
                logger.info(f"Neo4j GDS plugin available: v{version}")
                return True
        except Exception:
            pass
        logger.warning(
            "Neo4j GDS plugin NOT available. PageRank and Louvain will use pure Cypher fallbacks. To enable: add 'graph-data-science' plugin to Neo4j."
        )
        return False

    def run_community_detection(
        self, min_community_size: int = 3, write_results: bool = True
    ) -> List[CommunityResult]:
        if self._gds_available:
            return self._louvain_gds(min_community_size, write_results)
        else:
            return self._louvain_cypher_fallback(min_community_size)

    def _louvain_gds(
        self, min_community_size: int, write_results: bool
    ) -> List[CommunityResult]:
        graph_name = "fraud_account_graph"
        try:
            self.client.run_query(
                "\n                CALL gds.graph.project(\n                    $graph_name,\n                    'Account',\n                    {\n                        SENT_TO:        {orientation: 'UNDIRECTED'},\n                        USES:           {orientation: 'UNDIRECTED'},\n                        CONNECTED_FROM: {orientation: 'UNDIRECTED'}\n                    }\n                )\n                ",
                {"graph_name": graph_name},
                write=True,
            )
            logger.info(f"GDS graph projected: {graph_name}")
        except Exception as e:
            logger.warning(f"GDS graph projection failed: {e}")
            return self._louvain_cypher_fallback(min_community_size)
        try:
            self.client.run_query(
                "\n                CALL gds.louvain.write($graph_name, {\n                    writeProperty: 'community_id',\n                    maxIterations: 10,\n                    maxLevels: 5\n                })\n                YIELD communityCount, modularity\n                ",
                {"graph_name": graph_name},
                write=True,
            )
            logger.info("Louvain algorithm complete")
            return self._aggregate_communities(min_community_size)
        finally:
            try:
                self.client.run_query(
                    "CALL gds.graph.drop($graph_name)",
                    {"graph_name": graph_name},
                    write=True,
                )
            except Exception:
                pass

    def _louvain_cypher_fallback(
        self, min_community_size: int
    ) -> List[CommunityResult]:
        logger.info("Running Cypher-based community approximation (no GDS)")
        rows = self.client.run_query(
            "\n            MATCH (d:Device)<-[:USES]-(acc:Account)\n            WITH d.fingerprint AS device_fp, collect(acc) AS members\n            WHERE size(members) >= $min_size\n            UNWIND members AS acc\n            RETURN\n                device_fp                                  AS community_key,\n                acc.account_id                             AS account_id,\n                acc.risk_score                             AS risk_score,\n                acc.fraud_transaction_count                AS fraud_tx_count,\n                size(members)                              AS community_size\n            ORDER BY community_size DESC\n            ",
            {"min_size": min_community_size},
        )
        communities: Dict[str, CommunityResult] = {}
        for i, row in enumerate(rows):
            key = row["community_key"]
            if key not in communities:
                communities[key] = CommunityResult(
                    community_id=i, size=row.get("community_size", 0)
                )
            communities[key].account_ids.append(row["account_id"])
            if (row.get("fraud_tx_count") or 0) > 0:
                communities[key].fraud_account_count += 1
        return self._score_communities(list(communities.values()))

    def _aggregate_communities(self, min_community_size: int) -> List[CommunityResult]:
        rows = self.client.run_query(
            "\n            MATCH (acc:Account)\n            WHERE acc.community_id IS NOT NULL\n            WITH acc.community_id AS cid, collect(acc) AS members\n            WHERE size(members) >= $min_size\n            RETURN\n                cid                                                     AS community_id,\n                [m IN members | m.account_id]                          AS account_ids,\n                size(members)                                           AS size,\n                avg(coalesce(m.risk_score, 0.0))                       AS avg_risk,\n                size([m IN members WHERE m.fraud_transaction_count > 0]) AS fraud_count\n            ORDER BY size DESC\n            ",
            {"min_size": min_community_size},
        )
        results = []
        for row in rows:
            c = CommunityResult(
                community_id=row["community_id"],
                account_ids=row["account_ids"],
                size=row["size"],
                fraud_account_count=row["fraud_count"],
                avg_risk_score=round(row.get("avg_risk") or 0.0, 4),
            )
            results.append(c)
        return self._score_communities(results)

    def _score_communities(
        self, communities: List[CommunityResult]
    ) -> List[CommunityResult]:
        for c in communities:
            c.fraud_rate = (
                round(c.fraud_account_count / c.size, 4) if c.size > 0 else 0.0
            )
            if c.fraud_rate >= self.SUSPICIOUS_FRAUD_RATE_THRESHOLD:
                c.is_suspicious = True
                c.suspicion_reason = f"{c.fraud_account_count}/{c.size} members have fraud history ({c.fraud_rate:.0%} fraud rate)"
            elif c.avg_risk_score > 0.5:
                c.is_suspicious = True
                c.suspicion_reason = f"High avg risk score: {c.avg_risk_score:.2f}"
        suspicious_count = sum((1 for c in communities if c.is_suspicious))
        logger.info(
            f"Community detection complete | total={len(communities)} | suspicious={suspicious_count}"
        )
        return sorted(communities, key=lambda c: (-c.is_suspicious, -c.size))

    def run_pagerank(
        self, top_n: int = 100, write_results: bool = True
    ) -> List[PageRankResult]:
        if self._gds_available:
            return self._pagerank_gds(top_n, write_results)
        else:
            return self._pagerank_cypher_fallback(top_n)

    def _pagerank_gds(self, top_n: int, write_results: bool) -> List[PageRankResult]:
        graph_name = "fraud_pagerank_graph"
        try:
            self.client.run_query(
                "\n                CALL gds.graph.project(\n                    $graph_name,\n                    'Account',\n                    {SENT_TO: {orientation: 'NATURAL'}}\n                )\n                ",
                {"graph_name": graph_name},
                write=True,
            )
            if write_results:
                self.client.run_query(
                    "\n                    CALL gds.pageRank.write($graph_name, {\n                        writeProperty:   'pagerank_score',\n                        maxIterations:   20,\n                        dampingFactor:   0.85\n                    })\n                    YIELD nodePropertiesWritten, ranIterations\n                    ",
                    {"graph_name": graph_name},
                    write=True,
                )
            else:
                self.client.run_query(
                    "\n                    CALL gds.pageRank.stream($graph_name, {\n                        maxIterations: 20,\n                        dampingFactor: 0.85\n                    })\n                    YIELD nodeId, score\n                    ",
                    {"graph_name": graph_name},
                )
        finally:
            try:
                self.client.run_query(
                    "CALL gds.graph.drop($graph_name)",
                    {"graph_name": graph_name},
                    write=True,
                )
            except Exception:
                pass
        rows = self.client.run_query(
            "\n            MATCH (acc:Account)\n            WHERE acc.pagerank_score IS NOT NULL\n            RETURN\n                acc.account_id              AS account_id,\n                acc.pagerank_score          AS pagerank_score,\n                acc.fraud_transaction_count AS fraud_tx_count,\n                acc.risk_score              AS risk_score\n            ORDER BY acc.pagerank_score DESC\n            LIMIT $top_n\n            ",
            {"top_n": top_n},
        )
        return self._build_pagerank_results(rows)

    def _pagerank_cypher_fallback(self, top_n: int) -> List[PageRankResult]:
        logger.info("Running in-degree centrality fallback (no GDS PageRank)")
        rows = self.client.run_query(
            "\n            MATCH (acc:Account)\n            OPTIONAL MATCH (other:Account)-[:SENT_TO]->(acc)\n            WITH acc, count(other) AS in_degree\n            RETURN\n                acc.account_id              AS account_id,\n                toFloat(in_degree)          AS pagerank_score,\n                acc.fraud_transaction_count AS fraud_tx_count,\n                acc.risk_score              AS risk_score\n            ORDER BY in_degree DESC\n            LIMIT $top_n\n            ",
            {"top_n": top_n},
        )
        return self._build_pagerank_results(rows)

    def _build_pagerank_results(self, rows: List[Dict]) -> List[PageRankResult]:
        results = []
        for row in rows:
            score = float(row.get("pagerank_score") or 0.0)
            r = PageRankResult(
                account_id=row["account_id"],
                pagerank_score=round(score, 6),
                fraud_transaction_count=row.get("fraud_tx_count") or 0,
                risk_score=float(row.get("risk_score") or 0.0),
                is_hub=score >= self.PAGERANK_HUB_THRESHOLD,
            )
            results.append(r)
        hub_count = sum((1 for r in results if r.is_hub))
        logger.info(f"PageRank complete | top_n={len(results)} | hubs={hub_count}")
        return results

    def find_shortest_path(
        self, source_account_id: str, target_account_id: str, max_depth: int = 6
    ) -> ShortestPathResult:
        rows = self.client.run_query(
            f"\n            MATCH (source:Account {{account_id: $source_id}}),\n                  (target:Account {{account_id: $target_id}})\n            MATCH path = shortestPath(\n                (source)-[*1..{max_depth}]-(target)\n            )\n            RETURN\n                length(path)                                        AS path_length,\n                [n IN nodes(path) | coalesce(\n                    n.account_id,\n                    n.fingerprint,\n                    n.address,\n                    n.merchant_id,\n                    'unknown'\n                )]                                                  AS path_nodes,\n                [r IN relationships(path) | type(r)]               AS path_relationships\n            LIMIT 1\n            ",
            {"source_id": source_account_id, "target_id": target_account_id},
        )
        if not rows:
            logger.info(
                f"No path found between {source_account_id} and {target_account_id} within {max_depth} hops"
            )
            return ShortestPathResult(
                source_account=source_account_id,
                target_account=target_account_id,
                connected=False,
            )
        row = rows[0]
        result = ShortestPathResult(
            source_account=source_account_id,
            target_account=target_account_id,
            path_length=row.get("path_length", 0),
            path_nodes=row.get("path_nodes", []),
            path_relationships=row.get("path_relationships", []),
            connected=True,
        )
        logger.info(
            f"Path found: {source_account_id} -> {target_account_id} | hops={result.path_length} | via={'-> '.join(result.path_relationships)}"
        )
        return result

    def run_weakly_connected_components(
        self, min_size: int = 2
    ) -> List[Dict[str, Any]]:
        if self._gds_available:
            return self._wcc_gds(min_size)
        else:
            return self._wcc_cypher_fallback(min_size)

    def _wcc_gds(self, min_size: int) -> List[Dict[str, Any]]:
        graph_name = "fraud_wcc_graph"
        try:
            self.client.run_query(
                "\n                CALL gds.graph.project(\n                    $graph_name, 'Account',\n                    {SENT_TO: {orientation: 'UNDIRECTED'}}\n                )\n                ",
                {"graph_name": graph_name},
                write=True,
            )
            self.client.run_query(
                "\n                CALL gds.wcc.write($graph_name, {writeProperty: 'wcc_component_id'})\n                YIELD componentCount, componentDistribution\n                ",
                {"graph_name": graph_name},
                write=True,
            )
        finally:
            try:
                self.client.run_query(
                    "CALL gds.graph.drop($graph_name)",
                    {"graph_name": graph_name},
                    write=True,
                )
            except Exception:
                pass
        return self.client.run_query(
            "\n            MATCH (acc:Account)\n            WHERE acc.wcc_component_id IS NOT NULL\n            WITH acc.wcc_component_id AS comp_id, collect(acc) AS members\n            WHERE size(members) >= $min_size\n            RETURN\n                comp_id                                                       AS component_id,\n                size(members)                                                 AS size,\n                size([m IN members WHERE m.fraud_transaction_count > 0])      AS fraud_count,\n                avg(coalesce(m.risk_score, 0.0))                              AS avg_risk\n            ORDER BY fraud_count DESC\n            ",
            {"min_size": min_size},
        )

    def _wcc_cypher_fallback(self, min_size: int) -> List[Dict[str, Any]]:
        logger.info("Running shared-IP grouping as WCC fallback (no GDS)")
        return self.client.run_query(
            "\n            MATCH (ip:IPAddress)<-[:CONNECTED_FROM]-(acc:Account)\n            WITH ip.address AS ip_addr, collect(acc) AS members\n            WHERE size(members) >= $min_size\n            RETURN\n                ip_addr                                                       AS component_id,\n                size(members)                                                 AS size,\n                size([m IN members WHERE m.fraud_transaction_count > 0])      AS fraud_count,\n                avg(coalesce(m.risk_score, 0.0))                              AS avg_risk\n            ORDER BY fraud_count DESC\n            ",
            {"min_size": min_size},
        )

    def find_similar_accounts(
        self, account_id: str, top_n: int = 10, similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        rows = self.client.run_query(
            "\n            // Get all neighbors of the source account\n            MATCH (source:Account {account_id: $account_id})--(neighbor)\n            WITH source, collect(id(neighbor)) AS source_neighbors\n\n            // Find other accounts and their neighbors\n            MATCH (other:Account)--(other_neighbor)\n            WHERE other.account_id <> $account_id\n            WITH source, source_neighbors, other,\n                 collect(id(other_neighbor)) AS other_neighbors\n\n            // Compute Jaccard similarity\n            WITH source, other,\n                 source_neighbors, other_neighbors,\n                 [x IN source_neighbors WHERE x IN other_neighbors] AS intersection\n            WITH source, other,\n                 size(intersection)                                  AS shared_count,\n                 size(source_neighbors) + size(other_neighbors)\n                     - size(intersection)                            AS union_count,\n                 intersection\n            WHERE union_count > 0\n            WITH other,\n                 shared_count,\n                 toFloat(shared_count) / union_count                 AS similarity\n            WHERE similarity >= $threshold\n            RETURN\n                other.account_id              AS account_id,\n                round(similarity * 1000) / 1000 AS similarity,\n                shared_count                  AS shared_neighbors,\n                other.risk_score              AS risk_score,\n                other.fraud_transaction_count AS fraud_tx_count\n            ORDER BY similarity DESC\n            LIMIT $top_n\n            ",
            {
                "account_id": account_id,
                "threshold": similarity_threshold,
                "top_n": top_n,
            },
        )
        logger.info(
            f"Node similarity for {account_id}: found {len(rows)} similar accounts above threshold {similarity_threshold}"
        )
        return rows

    def run_full_fraud_ring_sweep(
        self, min_community_size: int = 3, suspicious_fraud_rate: float = 0.3
    ) -> Dict[str, Any]:
        logger.info("Starting full fraud ring sweep...")
        communities = self.run_community_detection(
            min_community_size=min_community_size
        )
        suspicious = [c for c in communities if c.is_suspicious]
        hubs = self.run_pagerank(top_n=50)
        hub_accounts = [r for r in hubs if r.is_hub]
        components = self.run_weakly_connected_components(min_size=min_community_size)
        high_risk_components = [
            c
            for c in components
            if c.get("fraud_count", 0) / max(c.get("size", 1), 1)
            >= suspicious_fraud_rate
        ]
        summary = {
            "communities_found": len(communities),
            "suspicious_communities": len(suspicious),
            "top_suspicious_communities": [
                {
                    "community_id": c.community_id,
                    "size": c.size,
                    "fraud_rate": c.fraud_rate,
                    "reason": c.suspicion_reason,
                    "member_count": len(c.account_ids),
                }
                for c in suspicious[:10]
            ],
            "hub_accounts": [r.account_id for r in hub_accounts],
            "hub_count": len(hub_accounts),
            "components_found": len(components),
            "high_risk_components": len(high_risk_components),
        }
        logger.info(
            f"Fraud ring sweep complete | suspicious_communities={len(suspicious)} | hubs={len(hub_accounts)} | high_risk_components={len(high_risk_components)}"
        )
        return summary
