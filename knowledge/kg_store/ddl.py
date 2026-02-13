"""PostgreSQL DDL for the Document Knowledge Graph (DKG).

Idempotent DDL (safe to run repeatedly).

Tables:
- kg_nodes: nodes keyed by (tenant, graph_version, node_id)
- kg_edges: edges keyed by (tenant, graph_version, edge_id)

node_id should be stable and canonical (e.g. "Concept:falha de autenticacao").
"""

DDL_STATEMENTS: list[str] = [
    # Optional extension for fuzzy name search
    "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
    # Nodes
    """
    CREATE TABLE IF NOT EXISTS kg_nodes (
        tenant TEXT NOT NULL,
        graph_version INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        node_type TEXT NOT NULL,
        name TEXT NOT NULL,
        aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
        properties JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (tenant, graph_version, node_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS kg_nodes_name_trgm_idx ON kg_nodes USING GIN (name gin_trgm_ops);",
    "CREATE INDEX IF NOT EXISTS kg_nodes_type_idx ON kg_nodes (tenant, graph_version, node_type);",
    # Edges
    """
    CREATE TABLE IF NOT EXISTS kg_edges (
        tenant TEXT NOT NULL,
        graph_version INTEGER NOT NULL,
        edge_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.5,
        evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (tenant, graph_version, edge_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS kg_edges_source_idx ON kg_edges (tenant, graph_version, source_id);",
    "CREATE INDEX IF NOT EXISTS kg_edges_target_idx ON kg_edges (tenant, graph_version, target_id);",
    "CREATE INDEX IF NOT EXISTS kg_edges_relation_idx ON kg_edges (tenant, graph_version, relation_type);",
    "CREATE INDEX IF NOT EXISTS kg_edges_evidence_gin_idx ON kg_edges USING GIN (evidence);",
]
