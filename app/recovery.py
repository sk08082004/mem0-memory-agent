def evaluate_relationship_recovery(
    user_graph,
    recovery,
    source_filter=None,
    attempt_limit=3,
):
    """Evaluate recovery eligibility for one validated user-scoped snapshot."""
    eligible = []
    details = []
    capped = []
    blocked = 0
    if not recovery["valid"]:
        return {
            "eligible": eligible,
            "details": details,
            "capped": capped,
            "blocked": recovery["blocked_count"],
            "invalid_records": recovery["invalid_records"],
        }

    graph_nodes = (
        user_graph.get("nodes", [])
        if isinstance(user_graph, dict)
        else []
    )
    node_ids = {
        node["id"]
        for node in graph_nodes
        if (
            isinstance(graph_nodes, list)
            and isinstance(node, dict)
            and isinstance(node.get("id"), str)
            and node["id"].strip()
        )
    } if isinstance(graph_nodes, list) else set()

    def endpoint_details(record, status, reason=None):
        record_source_id = record["source_id"]
        expected_edges = record.get("expected_edges", [])
        missing_target_ids = []
        if isinstance(expected_edges, list):
            missing_target_ids = sorted(
                {
                    edge["target_id"]
                    for edge in expected_edges
                    if (
                        isinstance(edge, dict)
                        and isinstance(edge.get("target_id"), str)
                        and edge["target_id"] not in node_ids
                    )
                }
            )
        detail = {
            "source_id": record_source_id,
            "status": status,
            "missing_source_ids": (
                [record_source_id]
                if record_source_id not in node_ids
                else []
            ),
            "missing_target_ids": missing_target_ids,
        }
        if reason:
            detail["reason"] = reason
        replacement_id = record.get("replacement_id")
        if isinstance(replacement_id, str) and replacement_id.strip():
            detail["replacement_id"] = replacement_id
        return detail

    for record in recovery["records"]:
        status = record["status"]
        record_source_id = record["source_id"]
        if source_filter is not None and source_filter != record_source_id:
            continue
        if status in {"succeeded", "succeeded_empty"}:
            continue
        if status == "superseded":
            details.append(
                endpoint_details(record, status, record.get("error"))
            )
            continue
        if status == "blocked_missing_endpoint":
            blocked += 1
            details.append(
                endpoint_details(record, status, record.get("error"))
            )
            continue
        if record_source_id not in node_ids:
            blocked += 1
            details.append(
                endpoint_details(
                    record,
                    "blocked_missing_endpoint",
                    "Source node is missing.",
                )
            )
            continue
        expected = record.get("expected_edges", [])
        missing_target = (
            status == "pending_persistence"
            and (
                not isinstance(expected, list)
                or any(
                    not isinstance(edge, dict)
                    or edge.get("target_id") not in node_ids
                    for edge in expected
                )
            )
        )
        if missing_target:
            blocked += 1
            details.append(
                endpoint_details(
                    record,
                    "blocked_missing_endpoint",
                    "One or more target nodes are missing.",
                )
            )
            continue
        if status not in {
            "pending_generation",
            "failed",
            "pending_persistence",
        }:
            blocked += 1
            details.append(
                {
                    "source_id": record_source_id,
                    "status": "blocked_invalid_record",
                    "reason": "Unsupported recovery status.",
                }
            )
            continue
        if (
            status in {"pending_generation", "failed"}
            and record["attempts"] >= attempt_limit
        ):
            capped.append(record_source_id)
            continue
        eligible.append(record)

    invalid_records = (
        [
            invalid_id
            for invalid_id in recovery["invalid_records"]
            if invalid_id == source_filter
        ]
        if source_filter is not None
        else recovery["invalid_records"]
    )
    return {
        "eligible": eligible,
        "details": details,
        "capped": capped,
        "blocked": blocked,
        "invalid_records": invalid_records,
    }
