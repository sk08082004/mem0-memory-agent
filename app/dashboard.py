import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph import KnowledgeGraph


GRAPH_FILE = "graph.json"
PENDING_STATUSES = {"pending_generation", "pending_persistence"}
FAILED_OR_BLOCKED_STATUSES = {"failed", "blocked_missing_endpoint"}
MAX_ERROR_DISPLAY_LENGTH = 160
MAX_MEMORY_PREVIEW_LENGTH = 120


def _safe_text_preview(value):
    if not isinstance(value, str) or not value.strip():
        return "No text available"

    preview = _redact_sensitive_text(value)
    preview = " ".join(preview.split())
    if len(preview) > MAX_MEMORY_PREVIEW_LENGTH:
        return preview[:MAX_MEMORY_PREVIEW_LENGTH - 1].rstrip() + "…"
    return preview


def _redact_sensitive_text(value):
    redacted = re.sub(
        r"(?i)(api[_ -]?key|secret|token|password|credential)"
        r"\s*[:=]\s*[\"']?[^\s,;\"']+",
        r"\1=[redacted]",
        value,
    )
    redacted = re.sub(
        r"(?i)\bbearer\s+\S+",
        "Bearer [redacted]",
        redacted,
    )
    return re.sub(
        r"\b(?:AIza|sk|m0|mem0|ghp|github_pat)[-_A-Za-z0-9]{16,}\b",
        "[redacted]",
        redacted,
        flags=re.IGNORECASE,
    )


def _safe_identifier(value):
    if not isinstance(value, str) or not value.strip():
        return "[missing or invalid]"
    return _redact_sensitive_text(value)[:128]


def _node_timestamp(node, key):
    value = node.get(key)
    if value is None and isinstance(node.get("metadata"), dict):
        value = node["metadata"].get(key)
    return value if isinstance(value, (str, int, float)) else None


class DashboardDataError(ValueError):
    """Raised when the persisted graph cannot be safely displayed."""


def load_dashboard_snapshot(graph_path, user_id):
    if not isinstance(user_id, str) or not user_id.strip():
        raise DashboardDataError("Enter a non-empty user ID.")

    selected_user = user_id.strip()
    path = Path(graph_path)
    if not path.exists():
        user_graph = None
    else:
        try:
            with path.open("r", encoding="utf-8") as graph_file:
                all_users_graph = json.load(graph_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DashboardDataError(
                f"Could not read the graph file: {error}"
            ) from error

        if not isinstance(all_users_graph, dict):
            raise DashboardDataError(
                "Graph file must contain a user-keyed JSON object."
            )
        user_graph = all_users_graph.get(selected_user)

    if user_graph is None:
        nodes = []
        relationships = []
        node_ids = set()
        recovery_queue = {}
    else:
        if (
            not isinstance(user_graph, dict)
            or not isinstance(user_graph.get("nodes"), list)
            or not isinstance(user_graph.get("relationships"), list)
        ):
            raise DashboardDataError(
                f"Graph data for user '{selected_user}' is malformed."
            )

        nodes = user_graph["nodes"]
        relationships = user_graph["relationships"]
        if any(
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not node["id"].strip()
            for node in nodes
        ):
            raise DashboardDataError(
                f"Graph nodes for user '{selected_user}' are malformed."
            )
        node_ids = {node["id"] for node in nodes}

        recovery_queue = user_graph.get(
            KnowledgeGraph.RELATIONSHIP_RECOVERY_KEY,
            {}
        )
        if not isinstance(recovery_queue, dict):
            raise DashboardDataError(
                f"Relationship recovery data for user '{selected_user}' "
                "is malformed."
            )

    recovery_records = []
    memory_records = []
    memory_texts = [
        node.get("memory_text")
        for node in nodes
        if isinstance(node.get("memory_text"), str) and node["memory_text"]
    ]
    for node in nodes:
        memory_records.append(
            {
                "node_id": _safe_identifier(node["id"]),
                "preview": _safe_text_preview(node.get("memory_text")),
                "created_at": _node_timestamp(node, "created_at") or "—",
                "updated_at": _node_timestamp(node, "updated_at") or "—",
            }
        )

    relationship_records = []
    for edge in relationships:
        if not isinstance(edge, dict):
            relationship_records.append(
                {
                    "source_id": "[invalid]",
                    "relationship": "[invalid relationship]",
                    "target_id": "[invalid]",
                }
            )
            continue

        source_id = edge.get("source")
        target_id = edge.get("target")
        relationship = edge.get("relationship")
        relationship_records.append(
            {
                "source_id": (
                    _safe_identifier(source_id)
                    if isinstance(source_id, str) and source_id in node_ids
                    else "[missing or invalid]"
                ),
                "relationship": (
                    _safe_text_preview(relationship)
                    if isinstance(relationship, str) and relationship.strip()
                    else "[missing or invalid]"
                ),
                "target_id": (
                    _safe_identifier(target_id)
                    if isinstance(target_id, str) and target_id in node_ids
                    else "[missing or invalid]"
                ),
            }
        )

    for source_id, record in recovery_queue.items():
        if (
            not KnowledgeGraph._valid_recovery_record(record)
            or source_id != record.get("source_id")
        ):
            recovery_records.append(
                {
                    "source_id": str(source_id),
                    "status": "invalid",
                    "attempts": "",
                    "error": "Malformed recovery record; inspect the graph file.",
                }
            )
            continue

        error = record.get("error") or ""
        for memory_text in memory_texts:
            error = error.replace(memory_text, "[memory text redacted]")
        error = _redact_sensitive_text(error)
        recovery_records.append(
            {
                "source_id": _safe_identifier(source_id),
                "status": record["status"],
                "attempts": record["attempts"],
                "error": error[:MAX_ERROR_DISPLAY_LENGTH],
            }
        )

    pending_count = sum(
        record["status"] in PENDING_STATUSES
        for record in recovery_records
    )
    failed_or_blocked_count = sum(
        record["status"] in FAILED_OR_BLOCKED_STATUSES
        or record["status"] == "invalid"
        for record in recovery_records
    )

    return {
        "user_id": selected_user,
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "pending_recovery_count": pending_count,
        "failed_or_blocked_count": failed_or_blocked_count,
        "memories": memory_records,
        "relationships": relationship_records,
        "recovery_records": recovery_records,
        "empty": not nodes and not relationships and not recovery_records,
    }


def main():
    import streamlit as st

    st.set_page_config(page_title="Memory Agent Status", layout="centered")
    st.title("Mem0 Memory Agent — Status")
    st.caption("Read-only local view of the persisted Knowledge Graph.")

    with st.form("dashboard_refresh"):
        user_id = st.text_input("User ID", key="dashboard_user_input")
        refresh = st.form_submit_button("Refresh")

    if refresh:
        st.session_state["dashboard_selected_user"] = user_id.strip()
    selected_user = st.session_state.get("dashboard_selected_user", "")
    if not selected_user:
        st.info("Enter a user ID to view its graph status.")
        return

    try:
        snapshot = load_dashboard_snapshot(GRAPH_FILE, selected_user)
    except DashboardDataError as error:
        st.error(str(error))
        return

    st.subheader(f"User: {snapshot['user_id']}")
    first, second = st.columns(2)
    first.metric("Graph nodes / memories", snapshot["node_count"])
    second.metric("Relationships", snapshot["relationship_count"])
    third, fourth = st.columns(2)
    third.metric("Pending recovery", snapshot["pending_recovery_count"])
    fourth.metric(
        "Failed or blocked",
        snapshot["failed_or_blocked_count"],
    )

    if snapshot["empty"]:
        st.info("No graph or relationship-recovery data exists for this user.")
    else:
        st.subheader("Stored memories")
        if snapshot["memories"]:
            st.dataframe(
                snapshot["memories"],
                hide_index=True,
                width="stretch",
                column_config={
                    "node_id": st.column_config.TextColumn(
                        "Memory ID",
                        width=160,
                    ),
                    "preview": st.column_config.TextColumn(
                        "Preview",
                        width=260,
                    ),
                    "created_at": st.column_config.TextColumn(
                        "Created",
                        width=120,
                    ),
                    "updated_at": st.column_config.TextColumn(
                        "Updated",
                        width=120,
                    ),
                },
                alt="Stored memories for the selected user",
            )
        else:
            st.info("This user has no stored graph nodes.")

        st.subheader("Relationships")
        if snapshot["relationships"]:
            st.dataframe(
                snapshot["relationships"],
                hide_index=True,
                width="stretch",
                column_config={
                    "source_id": st.column_config.TextColumn(
                        "Source",
                        width=180,
                    ),
                    "relationship": st.column_config.TextColumn(
                        "Relationship",
                        width=180,
                    ),
                    "target_id": st.column_config.TextColumn(
                        "Target",
                        width=180,
                    ),
                },
                alt="Knowledge graph relationships for the selected user",
            )
        else:
            st.info("This user has no graph relationships.")

    if snapshot["recovery_records"]:
        st.subheader("Relationship recovery")
        st.dataframe(
            snapshot["recovery_records"],
            hide_index=True,
            width="stretch",
            column_config={
                "source_id": st.column_config.TextColumn(
                    "Source",
                    width=160,
                ),
                "status": st.column_config.TextColumn(
                    "Status",
                    width=160,
                ),
                "attempts": st.column_config.TextColumn(
                    "Attempts",
                    width=100,
                ),
                "error": st.column_config.TextColumn(
                    "Error",
                    width=260,
                ),
            },
            alt="Relationship recovery status for the selected user",
        )
    else:
        st.info("This user has no relationship-recovery records.")


if __name__ == "__main__":
    main()
