from dataclasses import dataclass
from typing import List


NODE_LABELS = [
    "Condition",
    "EpisodeType",
    "Symptom",
    "Treatment",
    "Risk",
    "Trigger",
    "Comorbidity",
    "DiagnosticRule",
    "MonitoringRequirement",
    "Section",
    "Chunk",
]

EDGE_TYPES = [
    "HAS_EPISODE_TYPE",
    "HAS_SYMPTOM",
    "DIAGNOSED_BY",
    "DIFFERENTIATES_FROM",
    "TRIGGERED_BY",
    "CO_OCCURS_WITH",
    "TREATED_BY",
    "REQUIRES_MONITORING",
    "CONTRAINDICATED_OR_CAUTION",
    "INCREASES_RISK_OF",
    "REQUIRES_URGENT_REFERRAL",
    "SUPPORTED_BY_CHUNK",
]


@dataclass
class GraphSchemaDDL:
    constraints: List[str]
    indexes: List[str]


def build_schema_ddl() -> GraphSchemaDDL:
    constraints = [
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.section_key IS UNIQUE",
        "CREATE CONSTRAINT condition_name_unique IF NOT EXISTS FOR (n:Condition) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT episode_name_unique IF NOT EXISTS FOR (n:EpisodeType) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT treatment_name_unique IF NOT EXISTS FOR (n:Treatment) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT risk_name_unique IF NOT EXISTS FOR (n:Risk) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT trigger_name_unique IF NOT EXISTS FOR (n:Trigger) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT comorbidity_name_unique IF NOT EXISTS FOR (n:Comorbidity) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT rule_name_unique IF NOT EXISTS FOR (n:DiagnosticRule) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT monitor_name_unique IF NOT EXISTS FOR (n:MonitoringRequirement) REQUIRE n.name IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX chunk_section_idx IF NOT EXISTS FOR (c:Chunk) ON (c.section_id)",
        "CREATE INDEX chunk_lang_idx IF NOT EXISTS FOR (c:Chunk) ON (c.lang)",
    ]
    return GraphSchemaDDL(constraints=constraints, indexes=indexes)

