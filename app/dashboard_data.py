import csv
import io
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.graph import KnowledgeGraph
from app.recovery import evaluate_relationship_recovery


MAX_ERROR_DISPLAY_LENGTH = 160
MAX_MEMORY_PREVIEW_LENGTH = 120
GRAPH_MEMORY_PREVIEW_LENGTH = 54
DEFAULT_RECOVERY_ATTEMPT_LIMIT = 3
CSV_FIELDS = (
    "memory_id",
    "memory_text",
    "created_at",
    "updated_at",
    "recovery_status",
    "recovery_attempts",
    "relationship_count",
    "connectivity",
)


class DashboardDataError(ValueError):
    """Raised when the selected graph snapshot cannot be read safely."""


class DashboardSnapshotProvider:
    """Read-only snapshot interface for dashboard data providers."""

    def load_snapshot(self, user_id):
        raise NotImplementedError


class JsonGraphSnapshotProvider(DashboardSnapshotProvider):
    """Reads one user-scoped snapshot from the existing JSON graph store."""

    def __init__(self, graph_path):
        self.path = Path(graph_path).expanduser().resolve()

    def load_snapshot(self, user_id):
        selected_user = _validate_user_id(user_id)
        all_users_graph = _read_graph_file(self.path)
        user_graph = all_users_graph.get(selected_user)
        return normalize_dashboard_snapshot(
            user_graph,
            selected_user,
            self.path,
            _file_modified_time(self.path),
        )


def _validate_user_id(user_id):
    if not isinstance(user_id, str) or not user_id.strip():
        raise DashboardDataError("Enter a non-empty user ID.")
    return user_id.strip()


def _read_graph_file(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as graph_file:
            graph = json.load(graph_file)
    except FileNotFoundError:
        return {}
    except PermissionError as error:
        raise DashboardDataError(
            "Graph file is not readable (permission denied)."
        ) from error
    except UnicodeDecodeError as error:
        raise DashboardDataError(
            "Graph file is not readable (invalid text encoding)."
        ) from error
    except json.JSONDecodeError as error:
        raise DashboardDataError(
            "Could not read graph file (invalid JSON)."
        ) from error
    except OSError as error:
        raise DashboardDataError(
            "Graph file is not readable (filesystem error)."
        ) from error
    if not isinstance(graph, dict):
        raise DashboardDataError(
            "Graph file must contain a user-keyed JSON object."
        )
    return graph


def _file_modified_time(path):
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    except OSError:
        return None


def _redact_sensitive_text(value):
    redacted = re.sub(
        r"(?i)(api[_ -]?key|secret|token|password|credential)"
        r"\s*[:=]\s*[\"']?[^\s,;\"']+",
        r"\1=[redacted]",
        value,
    )
    redacted = re.sub(r"(?i)\bbearer\s+\S+", "******", redacted)
    return re.sub(
        r"\b(?:AIza|sk|m0|mem0|ghp|github_pat)[-_A-Za-z0-9]{16,}\b",
        "[redacted]",
        redacted,
        flags=re.IGNORECASE,
    )


def _safe_display_text(value):
    return _redact_sensitive_text(value) if isinstance(value, str) else ""


def _safe_identifier(value):
    if not isinstance(value, str) or not value.strip():
        return "[missing or invalid]"
    return _redact_sensitive_text(value)[:128]


def _parse_timestamp(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(
            normalized[:-1] + "+00:00"
            if normalized.endswith("Z")
            else normalized
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _node_timestamp(node, key):
    value = node.get(key)
    if value is None and isinstance(node.get("metadata"), dict):
        value = node["metadata"].get(key)
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return value if isinstance(value, str) else parsed.isoformat()


def _integrity_issue(code, severity, count, message):
    return {
        "code": code,
        "severity": severity,
        "count": count,
        "message": message,
    }


def normalize_dashboard_snapshot(
    user_graph,
    user_id,
    graph_path=None,
    graph_modified_at=None,
):
    issues = []
    if user_graph is None:
        user_graph = {"nodes": [], "relationships": []}
    if not isinstance(user_graph, dict):
        raise DashboardDataError("Graph data for the selected user is malformed.")

    raw_nodes = user_graph.get("nodes", [])
    raw_relationships = user_graph.get("relationships", [])
    if not isinstance(raw_nodes, list):
        raise DashboardDataError("Graph node collection is malformed.")
    if not isinstance(raw_relationships, list):
        raise DashboardDataError("Graph relationship collection is malformed.")

    nodes = []
    malformed_nodes = 0
    missing_text_count = 0
    missing_metadata_count = 0
    original_memory_texts = []
    for raw_node in raw_nodes:
        if (
            not isinstance(raw_node, dict)
            or not isinstance(raw_node.get("id"), str)
            or not raw_node["id"].strip()
        ):
            malformed_nodes += 1
            continue
        memory_text = raw_node.get("memory_text")
        if not isinstance(memory_text, str):
            memory_text = ""
            missing_text_count += 1
        original_memory_texts.append(memory_text)
        created_at = _node_timestamp(raw_node, "created_at")
        updated_at = _node_timestamp(raw_node, "updated_at")
        metadata = raw_node.get("metadata")
        if not isinstance(metadata, dict):
            missing_metadata_count += 1
        nodes.append(
            {
                "id": raw_node["id"],
                "display_id": _safe_identifier(raw_node["id"]),
                "memory_text": _safe_display_text(memory_text),
                "preview": _preview(memory_text),
                "created_at": created_at,
                "updated_at": updated_at,
                "created_sort": _parse_timestamp(created_at),
                "updated_sort": _parse_timestamp(updated_at),
                "metadata_present": isinstance(metadata, dict),
            }
        )
    if malformed_nodes:
        issues.append(
            _integrity_issue(
                "malformed_nodes",
                "error",
                malformed_nodes,
                "Some node entries are malformed and were excluded.",
            )
        )
    if missing_text_count:
        issues.append(
            _integrity_issue(
                "missing_memory_text",
                "warning",
                missing_text_count,
                "Some nodes have no usable memory text.",
            )
        )
    missing_created_at_count = sum(
        node["created_at"] is None for node in nodes
    )
    if missing_created_at_count:
        issues.append(
            _integrity_issue(
                "missing_creation_timestamps",
                "warning",
                missing_created_at_count,
                "Creation timestamps are unavailable for some memories.",
            )
        )
    missing_updated_at_count = sum(
        node["updated_at"] is None for node in nodes
    )
    if missing_updated_at_count:
        issues.append(
            _integrity_issue(
                "missing_update_timestamps",
                "warning",
                missing_updated_at_count,
                "Last-updated timestamps are unavailable for some memories.",
            )
        )
    if missing_metadata_count:
        issues.append(
            _integrity_issue(
                "missing_optional_metadata",
                "warning",
                missing_metadata_count,
                "Optional metadata is not present on some graph nodes.",
            )
        )

    ids = [node["id"] for node in nodes]
    id_counts = Counter(ids)
    duplicate_node_ids = sorted(
        node_id for node_id, count in id_counts.items() if count > 1
    )
    if duplicate_node_ids:
        issues.append(
            _integrity_issue(
                "duplicate_node_ids",
                "error",
                len(duplicate_node_ids),
                "Duplicate node IDs make graph endpoints ambiguous.",
            )
        )
    node_ids = set(ids)

    relationships = []
    relationship_records = []
    malformed_relationships = 0
    duplicate_relationships = 0
    missing_endpoint_relationships = 0
    relationship_keys = set()
    connected_ids = set()
    relationship_types = set()
    degrees = Counter()
    for raw_edge in raw_relationships:
        if not isinstance(raw_edge, dict):
            malformed_relationships += 1
            relationship_records.append(
                {
                    "source_id": "[missing or invalid]",
                    "relationship": "[invalid relationship]",
                    "target_id": "[missing or invalid]",
                }
            )
            continue
        source = raw_edge.get("source")
        target = raw_edge.get("target")
        relationship = raw_edge.get("relationship")
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(target, str)
            or not target.strip()
            or not isinstance(relationship, str)
            or not relationship.strip()
        ):
            malformed_relationships += 1
            relationship_records.append(
                {
                    "source_id": (
                        _safe_identifier(source)
                        if isinstance(source, str) and source in node_ids
                        else "[missing or invalid]"
                    ),
                    "relationship": (
                        _safe_display_text(relationship)
                        if isinstance(relationship, str)
                        and relationship.strip()
                        else "[missing or invalid]"
                    ),
                    "target_id": (
                        _safe_identifier(target)
                        if isinstance(target, str) and target in node_ids
                        else "[missing or invalid]"
                    ),
                }
            )
            continue
        relationship_records.append(
            {
                "source_id": (
                    _safe_identifier(source)
                    if source in node_ids
                    else "[missing or invalid]"
                ),
                "relationship": _safe_display_text(relationship),
                "target_id": (
                    _safe_identifier(target)
                    if target in node_ids
                    else "[missing or invalid]"
                ),
            }
        )
        key = (source, target, relationship)
        if key in relationship_keys:
            duplicate_relationships += 1
            continue
        relationship_keys.add(key)
        missing = source not in node_ids or target not in node_ids
        if missing:
            missing_endpoint_relationships += 1
        else:
            connected_ids.update((source, target))
            degrees[source] += 1
            degrees[target] += 1
        relationship_types.add(relationship)
        relationships.append(
            {
                "source": source,
                "source_display": _safe_identifier(source),
                "target": target,
                "target_display": _safe_identifier(target),
                "relationship": _safe_display_text(relationship),
                "missing_endpoint": missing,
            }
        )

    if malformed_relationships:
        issues.append(
            _integrity_issue(
                "malformed_relationships",
                "error",
                malformed_relationships,
                "Some relationship entries are malformed and were excluded.",
            )
        )
    if duplicate_relationships:
        issues.append(
            _integrity_issue(
                "duplicate_relationships",
                "error",
                duplicate_relationships,
                "Duplicate relationships were found.",
            )
        )
    if missing_endpoint_relationships:
        issues.append(
            _integrity_issue(
                "missing_relationship_endpoints",
                "error",
                missing_endpoint_relationships,
                "Relationships reference one or more nodes that do not exist.",
            )
        )

    recovery_queue = user_graph.get(
        KnowledgeGraph.RELATIONSHIP_RECOVERY_KEY,
        {},
    )
    if not isinstance(recovery_queue, dict):
        issues.append(
            _integrity_issue(
                "malformed_recovery_queue",
                "error",
                1,
                "The recovery queue is malformed and cannot be evaluated.",
            )
        )
        recovery = {
            "valid": False,
            "records": [],
            "invalid_records": [],
            "blocked_count": 1,
            "queue_size": 0,
            "error": "Relationship recovery queue is malformed.",
        }
    else:
        recovery_graph = {
            "nodes": raw_nodes,
            "relationships": raw_relationships,
            KnowledgeGraph.RELATIONSHIP_RECOVERY_KEY: recovery_queue,
        }
        recovery = KnowledgeGraph.validate_relationship_recovery(
            recovery_graph
        )
    if not recovery["valid"]:
        issues.append(
            _integrity_issue(
                "invalid_recovery_queue",
                "error",
                recovery["blocked_count"] or 1,
                "Recovery records could not be safely evaluated.",
            )
        )

    invalid_recovery_ids = set(recovery["invalid_records"])
    recovery_records = []
    for source_id, record in recovery_queue.items() if isinstance(
        recovery_queue,
        dict,
    ) else []:
        valid = (
            KnowledgeGraph._valid_recovery_record(record)
            and source_id == record.get("source_id")
        )
        if not valid:
            invalid_recovery_ids.add(str(source_id))
            invalid_recovery = (
                record if isinstance(record, dict) else {}
            )
            recovery_records.append(
                {
                    "source_id": source_id,
                    "display_source_id": _safe_identifier(source_id),
                    "status": "invalid",
                    "attempts": None,
                    "attempt_cap": KnowledgeGraph.MAX_RECOVERY_ATTEMPTS,
                    "eligible": False,
                    "eligibility_reason": "invalid_recovery_record",
                    "replacement_id": (
                        _safe_identifier(invalid_recovery["replacement_id"])
                        if isinstance(
                            invalid_recovery.get("replacement_id"),
                            str,
                        )
                        else None
                    ),
                    "error": "Malformed recovery record.",
                    "valid": False,
                }
            )
            continue
        error = record.get("error") or ""
        for memory_text in original_memory_texts:
            if memory_text:
                error = error.replace(memory_text, "[memory text redacted]")
        error = _redact_sensitive_text(error)[:MAX_ERROR_DISPLAY_LENGTH]
        recovery_records.append(
            {
                "source_id": source_id,
                "display_source_id": _safe_identifier(source_id),
                "status": record["status"],
                "attempts": record["attempts"],
                "attempt_cap": KnowledgeGraph.MAX_RECOVERY_ATTEMPTS,
                "eligible": False,
                "eligibility_reason": "not_evaluated",
                "replacement_id": (
                    _safe_identifier(record["replacement_id"])
                    if isinstance(record.get("replacement_id"), str)
                    else None
                ),
                "error": error,
                "valid": True,
            }
        )

    attempt_limit = min(
        DEFAULT_RECOVERY_ATTEMPT_LIMIT,
        KnowledgeGraph.MAX_RECOVERY_ATTEMPTS,
    )
    eligibility = evaluate_relationship_recovery(
        {
            "nodes": raw_nodes,
            "relationships": raw_relationships,
        },
        recovery,
        attempt_limit=attempt_limit,
    )
    eligible_ids = {
        record["source_id"] for record in eligibility["eligible"]
    }
    capped_ids = set(eligibility["capped"])
    detail_by_id = {
        detail["source_id"]: detail
        for detail in eligibility["details"]
        if "source_id" in detail
    }
    for record in recovery_records:
        source_id = record.get("source_id")
        if not record["valid"]:
            continue
        record["eligible"] = source_id in eligible_ids
        if record["eligible"]:
            record["eligibility_reason"] = "eligible"
        elif source_id in capped_ids:
            record["eligibility_reason"] = "attempt_cap_reached"
        elif source_id in detail_by_id:
            detail = detail_by_id[source_id]
            if detail.get("missing_target_ids"):
                record["eligibility_reason"] = "target_node_missing"
            elif detail.get("missing_source_ids"):
                record["eligibility_reason"] = "source_node_missing"
            else:
                record["eligibility_reason"] = detail.get(
                    "reason",
                    detail["status"],
                )
        elif source_id in {"", None}:
            record["eligibility_reason"] = "invalid_source_id"
        elif record["status"] in {
            "succeeded",
            "succeeded_empty",
            "superseded",
        }:
            record["eligibility_reason"] = "terminal_status"
        else:
            record["eligibility_reason"] = "not_eligible"

    if invalid_recovery_ids:
        issues.append(
            _integrity_issue(
                "invalid_recovery_records",
                "error",
                len(invalid_recovery_ids),
                "Malformed or mismatched recovery records were found.",
            )
        )
    if recovery_records:
        issues.append(
            _integrity_issue(
                "transition_history_unavailable",
                "warning",
                len(recovery_records),
                "Transition history is not stored, so prior status transitions "
                "cannot be verified.",
            )
        )

    status_counts = Counter()
    for record in recovery_records:
        status_counts[record["status"]] += 1

    memory_relationships = Counter()
    for edge in relationships:
        if not edge["missing_endpoint"]:
            memory_relationships[edge["source"]] += 1
            memory_relationships[edge["target"]] += 1
    recovery_by_id = {
        record.get("source_id"): record
        for record in recovery_records
        if record.get("source_id") is not None
    }
    for node in nodes:
        node["relationship_count"] = memory_relationships[node["id"]]
        node["connectivity"] = (
            "connected" if node["id"] in connected_ids else "isolated"
        )
        recovery_record = recovery_by_id.get(node["id"])
        node["recovery"] = (
            {
                "status": recovery_record["status"],
                "attempts": recovery_record["attempts"],
                "attempt_cap": recovery_record["attempt_cap"],
                "eligible": recovery_record["eligible"],
                "eligibility_reason": recovery_record["eligibility_reason"],
                "error": recovery_record["error"],
                "replacement_id": recovery_record["replacement_id"],
            }
            if recovery_record
            else None
        )
    _assign_graph_labels(nodes)

    timeline = Counter()
    for node in nodes:
        created = node["created_sort"]
        if created is not None:
            timeline[created.date().isoformat()] += 1

    valid_node_count = len(nodes)
    valid_relationship_count = len(relationships)
    error_count = sum(
        issue["count"] for issue in issues if issue["severity"] == "error"
    )
    warning_count = sum(
        issue["count"] for issue in issues if issue["severity"] == "warning"
    )
    return {
        "user_id": user_id,
        "graph_path": str(graph_path) if graph_path else None,
        "graph_modified_at": graph_modified_at,
        "node_count": valid_node_count,
        "raw_node_count": len(raw_nodes),
        "relationship_count": valid_relationship_count,
        "raw_relationship_count": len(raw_relationships),
        "nodes_with_created_at_count": sum(
            node["created_at"] is not None for node in nodes
        ),
        "nodes_missing_created_at_count": sum(
            node["created_at"] is None for node in nodes
        ),
        "nodes_missing_updated_at_count": sum(
            node["updated_at"] is None for node in nodes
        ),
        "failed_or_blocked_count": (
            status_counts["failed"]
            + status_counts["blocked_missing_endpoint"]
            + status_counts["invalid"]
        ),
        "connected_node_count": sum(
            node["connectivity"] == "connected" for node in nodes
        ),
        "isolated_node_count": sum(
            node["connectivity"] == "isolated" for node in nodes
        ),
        "recovery_record_count": (
            len(recovery_queue) if isinstance(recovery_queue, dict) else 0
        ),
        "valid_recovery_record_count": sum(
            record["valid"] for record in recovery_records
        ),
        "pending_recovery_count": sum(
            status_counts[status]
            for status in ("pending_generation", "pending_persistence")
        ),
        "failed_recovery_count": status_counts["failed"],
        "succeeded_recovery_count": sum(
            status_counts[status]
            for status in ("succeeded", "succeeded_empty")
        ),
        "blocked_recovery_count": status_counts["blocked_missing_endpoint"],
        "superseded_recovery_count": status_counts["superseded"],
        "eligible_recovery_count": sum(
            record["eligible"] for record in recovery_records
        ),
        "capped_recovery_count": sum(
            record["eligibility_reason"] == "attempt_cap_reached"
            for record in recovery_records
        ),
        "recovery_status_counts": dict(status_counts),
        "memories": [
            {
                "node_id": node["display_id"],
                "preview": node["preview"],
                "created_at": node["created_at"] or "—",
                "updated_at": node["updated_at"] or "—",
            }
            for node in nodes
        ],
        "memory_items": nodes,
        "relationships": relationship_records,
        "relationship_items": relationships,
        "recovery_records": recovery_records,
        "integrity": {
            "errors": [issue for issue in issues if issue["severity"] == "error"],
            "warnings": [
                issue for issue in issues if issue["severity"] == "warning"
            ],
            "error_count": error_count,
            "warning_count": warning_count,
            "issues": issues,
        },
        "relationship_types": sorted(relationship_types),
        "timeline": [
            {"date": date, "memories": count}
            for date, count in sorted(timeline.items())
        ],
        "eligible_source_ids": sorted(eligible_ids),
        "capped_source_ids": sorted(capped_ids),
        "empty": (
            not valid_node_count
            and not valid_relationship_count
            and not recovery_records
        ),
    }


def _preview(value):
    if not isinstance(value, str) or not value.strip():
        return "No text available"
    preview = " ".join(_safe_display_text(value).split())
    if len(preview) > MAX_MEMORY_PREVIEW_LENGTH:
        return preview[:MAX_MEMORY_PREVIEW_LENGTH - 1].rstrip() + "…"
    return preview


def _graph_preview(value):
    if not isinstance(value, str) or not value.strip():
        return "Memory text unavailable"
    preview = " ".join(_safe_display_text(value).split())
    if len(preview) > GRAPH_MEMORY_PREVIEW_LENGTH:
        return preview[:GRAPH_MEMORY_PREVIEW_LENGTH - 1].rstrip() + "…"
    return preview


def _preview_collision_key(value):
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(normalized.split())[:42]


def _short_id(value):
    safe_id = _safe_identifier(value)
    return safe_id[-6:] if safe_id != "[missing or invalid]" else "unknown"


def _assign_graph_labels(memories):
    groups = {}
    for memory in memories:
        memory["graph_preview"] = _graph_preview(memory["memory_text"])
        key = _preview_collision_key(memory["graph_preview"])
        groups.setdefault(key, []).append(memory)

    for group in groups.values():
        collision = len(group) > 1
        used_labels = set()
        for index, memory in enumerate(group, start=1):
            suffix = f" · {_short_id(memory['id'])}" if collision else ""
            label = f"{memory['graph_preview']}{suffix}"
            if label in used_labels:
                label = f"{label}-{index}"
            used_labels.add(label)
            memory["graph_label"] = label
            memory["short_id"] = _short_id(memory["id"])


def load_dashboard_snapshot(graph_path, user_id):
    return JsonGraphSnapshotProvider(graph_path).load_snapshot(user_id)


def filter_memory_items(
    memories,
    query="",
    recovery_statuses=None,
    connectivity="All",
    sort_by="Most recently updated",
    descending=True,
):
    statuses = set(recovery_statuses or [])
    query = query.casefold().strip()
    filtered = []
    for memory in memories:
        recovery = memory.get("recovery") or {}
        status = recovery.get("status") or "untracked"
        if statuses and status not in statuses:
            continue
        if connectivity != "All" and memory.get("connectivity") != connectivity:
            continue
        searchable = " ".join(
            (
                str(memory.get("id", "")),
                str(memory.get("memory_text", "")),
                str(status),
            )
        ).casefold()
        if query and query not in searchable:
            continue
        filtered.append(memory)

    sort_key = {
        "Most recently created": "created_sort",
        "Most recently updated": "updated_sort",
        "Memory ID": "id",
        "Most connected": "relationship_count",
    }.get(sort_by, "updated_sort")
    if sort_key in {"created_sort", "updated_sort"}:
        filtered.sort(
            key=lambda item: item.get(sort_key) or datetime.min.replace(
                tzinfo=timezone.utc
            ),
            reverse=descending,
        )
    elif sort_key == "relationship_count":
        filtered.sort(
            key=lambda item: item.get(sort_key, 0),
            reverse=descending,
        )
    else:
        filtered.sort(
            key=lambda item: str(item.get(sort_key, "")).casefold(),
            reverse=descending,
        )
    return filtered


def paginate(items, page, page_size):
    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive integer.")
    if type(page_size) is not int or page_size < 1:
        raise ValueError("page_size must be a positive integer.")
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return items[start:start + page_size], total_pages


def export_memories_csv(memories, include_memory_text=False):
    fields = list(CSV_FIELDS)
    if not include_memory_text:
        fields.remove("memory_text")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for memory in memories:
        recovery = memory.get("recovery") or {}
        row = {
            "memory_id": _csv_safe(_safe_identifier(memory.get("id"))),
            "memory_text": _csv_safe(
                _safe_display_text(memory.get("memory_text"))
            ),
            "created_at": memory.get("created_at") or "",
            "updated_at": memory.get("updated_at") or "",
            "recovery_status": recovery.get("status") or "untracked",
            "recovery_attempts": recovery.get("attempts", ""),
            "relationship_count": memory.get("relationship_count", 0),
            "connectivity": memory.get("connectivity", "isolated"),
        }
        for key in (
            "created_at",
            "updated_at",
            "recovery_status",
            "recovery_attempts",
            "connectivity",
        ):
            row[key] = _csv_safe(row[key])
        writer.writerow(row)
    return output.getvalue()


def _csv_safe(value):
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def build_graph_mermaid(memories, relationships, selected_node_id=None):
    node_ids = [memory["id"] for memory in memories]
    available = set(node_ids)
    memories_by_id = {memory["id"]: memory for memory in memories}
    connected = {
        memory["id"]
        for memory in memories
        if memory.get("connectivity") == "connected"
    }
    selected_neighbors = {
        endpoint
        for edge in relationships
        if selected_node_id
        and selected_node_id in (edge["source"], edge["target"])
        for endpoint in (edge["source"], edge["target"])
    }
    lines = [
        "flowchart LR",
        "classDef selected fill:#2563eb,color:#ffffff,stroke:#1d4ed8,stroke-width:3px",
        "classDef neighbor fill:#bfdbfe,color:#172554,stroke:#60a5fa",
        "classDef isolated fill:#e5e7eb,color:#374151,stroke:#9ca3af",
        "classDef connected fill:#d1fae5,color:#064e3b,stroke:#6ee7b7",
    ]
    for index, node_id in enumerate(node_ids):
        safe_node = f"n{index}"
        memory = memories_by_id[node_id]
        label = _wrap_graph_label(
            _mermaid_escape(memory["graph_label"])
        )
        if selected_node_id == node_id:
            class_name = "selected"
        elif node_id in selected_neighbors:
            class_name = "neighbor"
        elif node_id not in connected:
            class_name = "isolated"
        else:
            class_name = "connected"
        lines.append(f'  {safe_node}["{label}"]:::{class_name}')
    index_by_id = {node_id: f"n{index}" for index, node_id in enumerate(node_ids)}
    for edge in relationships:
        if (
            edge.get("missing_endpoint")
            or edge["source"] not in available
            or edge["target"] not in available
        ):
            continue
        label = _mermaid_escape(edge["relationship"])
        lines.append(
            f'  {index_by_id[edge["source"]]} -->|{label}| '
            f'{index_by_id[edge["target"]]}'
        )
    return "\n".join(lines)


def _wrap_graph_label(value, width=24):
    words = value.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "<br/>".join(lines)


def graph_subset(memories, relationships, relationship_types=None,
                 selected_node_id=None, max_nodes=50):
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer.")
    allowed_types = set(relationship_types or [])
    filtered_edges = [
        edge
        for edge in relationships
        if not edge.get("missing_endpoint")
        and (not allowed_types or edge["relationship"] in allowed_types)
    ]
    nodes_by_id = {memory["id"]: memory for memory in memories}
    candidates = []
    if selected_node_id in nodes_by_id:
        candidates.append(selected_node_id)
        for edge in filtered_edges:
            if selected_node_id == edge["source"]:
                candidates.append(edge["target"])
            elif selected_node_id == edge["target"]:
                candidates.append(edge["source"])
    candidates.extend(memory["id"] for memory in memories)
    selected_ids = []
    seen = set()
    for node_id in candidates:
        if node_id in nodes_by_id and node_id not in seen:
            selected_ids.append(node_id)
            seen.add(node_id)
        if len(selected_ids) >= max_nodes:
            break
    selected_set = set(selected_ids)
    return (
        [nodes_by_id[node_id] for node_id in selected_ids],
        [
            edge
            for edge in filtered_edges
            if edge["source"] in selected_set
            and edge["target"] in selected_set
        ],
    )


def _mermaid_escape(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("\n", " ")
    )


def graph_freshness_label(modified_at):
    parsed = _parse_timestamp(modified_at)
    if parsed is None:
        return "Modification time unavailable"
    age_seconds = max(
        0,
        (datetime.now(timezone.utc) - parsed).total_seconds(),
    )
    if age_seconds < 60:
        return "Modified less than a minute ago"
    if age_seconds < 3600:
        return f"Modified {int(age_seconds // 60)} minutes ago"
    if age_seconds < 86400:
        return f"Modified {int(age_seconds // 3600)} hours ago"
    return f"Modified {int(age_seconds // 86400)} days ago"
