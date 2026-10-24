import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard_data import (
    DashboardDataError,
    build_graph_mermaid,
    export_memories_csv,
    filter_memory_items,
    graph_freshness_label,
    graph_subset,
    load_dashboard_snapshot,
    paginate,
)


GRAPH_FILE = "graph.json"
RECOVERY_STATUSES = (
    "pending_generation",
    "pending_persistence",
    "failed",
    "succeeded",
    "succeeded_empty",
    "blocked_missing_endpoint",
    "superseded",
    "invalid",
)
MAX_GRAPH_NODES = 150
INTEGRITY_GUIDANCE = {
    "duplicate_node_ids": (
        "Memory IDs are not unique, so a relationship may be ambiguous.",
        "Memory lookup and connection display may be unreliable.",
        "Review duplicate IDs in the persisted graph before using graph recovery.",
    ),
    "duplicate_relationships": (
        "The same source, target, and relationship is stored more than once.",
        "Relationship counts and graph rendering may be misleading.",
        "Review the duplicate relationship records; do not auto-repair from this dashboard.",
    ),
    "missing_relationship_endpoints": (
        "A stored connection references a memory that is absent from this graph.",
        "The connection is excluded from valid connectivity and graph rendering.",
        "Inspect the referenced source and target IDs through the operator workflow.",
    ),
    "malformed_nodes": (
        "Some graph node entries do not have the required structure or ID.",
        "Malformed nodes are excluded from memory details and connectivity.",
        "Review the affected node entries before relying on graph totals.",
    ),
    "malformed_relationships": (
        "Some relationship entries are missing valid endpoints or a relationship type.",
        "These entries cannot be interpreted as graph connections.",
        "Review the affected relationship entries in the persisted graph.",
    ),
    "invalid_recovery_queue": (
        "The recovery queue is malformed or exceeds its validation limits.",
        "Recovery eligibility cannot be determined safely.",
        "Use the existing CLI/operator procedure to inspect the queue; do not retry jobs from this page.",
    ),
    "invalid_recovery_records": (
        "One or more recovery records failed schema or queue-key validation.",
        "These records are not considered eligible for recovery.",
        "Inspect the malformed records through the CLI without enrolling or retrying automatically.",
    ),
    "missing_creation_timestamps": (
        "Creation dates are not stored for some memories.",
        "The creation timeline and chronological sorting are incomplete.",
        "Persist creation timestamps for new nodes; historical dates cannot be inferred safely.",
    ),
    "missing_update_timestamps": (
        "Last-updated dates are not stored for some memories.",
        "Sorting by most recently updated is unavailable for those records.",
        "Persist update timestamps when graph nodes change.",
    ),
    "missing_optional_metadata": (
        "Optional metadata is absent for some memories.",
        "Metadata-based context may be unavailable, but core memory text remains usable.",
        "Add metadata only where the application has a reliable source for it.",
    ),
    "transition_history_unavailable": (
        "Only the current recovery status is stored, not its event history.",
        "Previous status transitions cannot be checked retrospectively.",
        "Persist recovery status events in a future audit store if transition history is required.",
    ),
}
RECOVERY_STATUS_LABELS = {
    "pending_generation": "Waiting for recovery",
    "pending_persistence": "Waiting to save relationships",
    "failed": "Failed; inspect error details",
    "succeeded": "Completed",
    "succeeded_empty": "Completed; no new relationships found",
    "blocked_missing_endpoint": "Blocked; a referenced memory is missing",
    "superseded": "Replaced by another memory",
    "invalid": "Invalid recovery record",
}
ELIGIBILITY_LABELS = {
    "eligible": "Ready for recovery",
    "attempt_cap_reached": "Attempt limit reached",
    "target_node_missing": "A related memory is missing",
    "source_node_missing": "Source memory is missing",
    "invalid_recovery_record": "Recovery details are invalid",
    "invalid_recovery_queue": "Recovery queue cannot be validated",
    "recovery_record_not_found": "No recovery record exists",
    "record_not_eligible": "This record is already complete",
    "terminal_status": "Historical record; no action needed",
    "not_eligible": "Not currently eligible",
}


def _stat_key(graph_path):
    try:
        details = os.stat(graph_path)
        return details.st_mtime_ns, details.st_size
    except OSError:
        return None, None


def _status_label(status):
    return RECOVERY_STATUS_LABELS.get(
        status,
        status.replace("_", " ").capitalize(),
    )


def _format_timestamp(value):
    if not value or value == "—":
        return "Unavailable"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "Invalid timestamp"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _eligibility_label(reason):
    return ELIGIBILITY_LABELS.get(
        reason,
        str(reason).replace("_", " ").capitalize(),
    )


def _select_connected_graph_memory():
    import streamlit as st

    connected_id = st.session_state.get("dashboard_graph_connected_node")
    if connected_id:
        st.session_state["dashboard_graph_selected_node"] = connected_id


def _memory_rows(memories):
    return [
        {
            "Memory": memory["preview"],
            "Memory ID": memory["display_id"],
            "Created": _format_timestamp(memory["created_at"]),
            "Last updated": _format_timestamp(memory["updated_at"]),
            "Recovery status": _status_label(
                (memory.get("recovery") or {}).get(
                    "status",
                    "untracked",
                )
            ),
            "Attempts": (
                str((memory.get("recovery") or {}).get("attempts"))
                if memory.get("recovery")
                and (memory.get("recovery") or {}).get("attempts") is not None
                else "—"
            ),
            "Connectivity": memory["connectivity"],
            "Relationships": memory["relationship_count"],
        }
        for memory in memories
    ]


def _recovery_rows(snapshot):
    memories_by_id = {
        memory["id"]: memory for memory in snapshot["memory_items"]
    }
    rows = []
    for record in snapshot["recovery_records"]:
        source_id = record.get("source_id")
        source = memories_by_id.get(source_id)
        replacement = memories_by_id.get(record.get("replacement_id"))
        rows.append(
            {
                "Memory": (
                    source["preview"] if source else "Memory text unavailable"
                ),
                "Status": _status_label(record["status"]),
                "Attempts": (
                    f"{record['attempts']} / {record['attempt_cap']}"
                    if record["attempts"] is not None
                    else "Unavailable"
                ),
                "Source node": (
                    "Present" if source else "Missing"
                ),
                "Eligible": (
                    "Yes" if record["eligible"] else "No"
                ),
                "Reason": _eligibility_label(
                    record["eligibility_reason"]
                ),
                "Replacement memory": (
                    replacement["preview"]
                    if replacement
                    else (
                        "Replacement node unavailable"
                        if record.get("replacement_id")
                        else "—"
                    )
                ),
                "Error details": record.get("error") or "—",
                "Memory ID": record.get(
                    "display_source_id",
                    "[invalid]",
                ),
                "Technical status": record.get("status", "invalid"),
                "Replacement ID": record.get("replacement_id") or "—",
            }
        )
    return rows


def _recovery_attention_action(record):
    status = record["status"]
    if status == "failed":
        return (
            "Review the sanitized error and attempt count; use the CLI only "
            "after confirming the source and its context."
        )
    if status == "blocked_missing_endpoint":
        return (
            "Check whether the source or an expected target memory is missing "
            "from this user's graph."
        )
    if status == "pending_persistence":
        return (
            "Inspect the pending relationship targets and confirm their nodes "
            "exist before using the CLI."
        )
    if record["eligibility_reason"] == "attempt_cap_reached":
        return (
            "Review prior errors and the source memory before considering "
            "any further recovery through the CLI."
        )
    if not record["valid"]:
        return (
            "Inspect the malformed recovery record using the existing "
            "operator workflow; the dashboard will not repair it."
        )
    return "Review the source memory and recovery details in the CLI."


def _integrity_rows(issues):
    rows = []
    for issue in issues:
        explanation, impact, next_step = INTEGRITY_GUIDANCE.get(
            issue["code"],
            (
                issue["message"],
                "Some dashboard details may be incomplete.",
                "Review this finding with the existing read-only diagnostics.",
            ),
        )
        affected = (
            "Current recovery records"
            if issue["code"] == "transition_history_unavailable"
            else f"{issue['count']:,}"
        )
        rows.append(
            {
                "Issue": issue["code"].replace("_", " ").capitalize(),
                "Severity": issue["severity"].capitalize(),
                "Affected records": affected,
                "Explanation": explanation,
                "Impact": impact,
                "Recommended next step": next_step,
            }
        )
    return rows


def _connected_memories(snapshot, memory_id):
    memories_by_id = {
        memory["id"]: memory for memory in snapshot["memory_items"]
    }
    related = []
    for edge in snapshot["relationship_items"]:
        source = edge["source"]
        target = edge["target"]
        if memory_id not in (source, target):
            continue
        if source == memory_id:
            connected_id = target
            direction = "Outgoing"
        else:
            connected_id = source
            direction = "Incoming"
        connected_memory = memories_by_id.get(connected_id)
        related.append(
            {
                "memory_id": connected_id if connected_memory else None,
                "memory": connected_memory,
                "preview": (
                    connected_memory["preview"]
                    if connected_memory
                    else "Missing memory endpoint"
                ),
                "memory_id_display": (
                    connected_memory["display_id"]
                    if connected_memory
                    else "[missing or invalid]"
                ),
                "relationship": edge["relationship"],
                "direction": direction,
                "missing_endpoint": not connected_memory,
            }
        )
    return related


def _render_overview(st, snapshot):
    st.header("Overview", icon=":material/dashboard:")
    with st.container(horizontal=True):
        st.metric(
            "Memories stored",
            snapshot["node_count"],
            help="Valid memory nodes in this user's selected graph.",
            border=True,
        )
        st.metric("Relationships", snapshot["relationship_count"], border=True)
        st.metric(
            "Connected memories",
            snapshot["connected_node_count"],
            help="Memories with at least one valid relationship endpoint.",
            border=True,
        )
        st.metric(
            "Isolated memories",
            snapshot["isolated_node_count"],
            help="Memories that have no valid graph relationships.",
            border=True,
        )
    st.caption(
        "Connected and isolated counts are based only on relationships whose "
        "source and target nodes both exist."
    )
    with st.container(horizontal=True):
        st.metric(
            "Recovery records",
            snapshot["recovery_record_count"],
            border=True,
        )
        st.metric(
            "Pending",
            snapshot["pending_recovery_count"],
            border=True,
        )
        st.metric("Failed", snapshot["failed_recovery_count"], border=True)
        st.metric(
            "Succeeded",
            snapshot["succeeded_recovery_count"],
            border=True,
        )
        st.metric(
            "Blocked",
            snapshot["blocked_recovery_count"],
            border=True,
        )
        st.metric(
            "Integrity errors",
            snapshot["integrity"]["error_count"],
            border=True,
        )
    if snapshot["failed_recovery_count"] or snapshot["blocked_recovery_count"]:
        st.warning(
            "Recovery needs attention. Review failed records and missing "
            "relationship endpoints in Recovery centre.",
            icon=":material/warning:",
        )
    elif snapshot["capped_recovery_count"]:
        st.warning(
            "Some recovery jobs reached the attempt limit. Review their "
            "error details before taking action through the CLI.",
            icon=":material/warning:",
        )
    elif snapshot["pending_recovery_count"]:
        st.info(
            "Recovery work is waiting or has relationship data to persist. "
            "The dashboard is read-only; inspect the job in Recovery centre.",
            icon=":material/schedule:",
        )
    else:
        st.success("No unresolved recovery jobs were detected.")
    if snapshot["integrity"]["error_count"]:
        st.warning(
            "Graph integrity errors were detected. Review the Data integrity "
            "section before relying on graph connections."
        )

    status_column, timeline_column = st.columns(2)
    with status_column:
        with st.container(border=True):
            st.subheader("Recovery status")
            status_counts = snapshot["recovery_status_counts"]
            if status_counts:
                import altair as alt
                import pandas as pd

                status_colors = {
                    "Waiting for recovery": "#f59e0b",
                    "Waiting to save relationships": "#eab308",
                    "Failed; inspect error details": "#ef4444",
                    "Completed": "#22c55e",
                    "Completed; no new relationships found": "#10b981",
                    "Blocked; a referenced memory is missing": "#f97316",
                    "Replaced by another memory": "#64748b",
                    "Invalid recovery record": "#dc2626",
                }
                status_data = pd.DataFrame(
                    [
                        {
                            "Status": _status_label(status),
                            "Records": status_counts[status],
                            "Color": status_colors.get(
                                _status_label(status),
                                "#64748b",
                            ),
                        }
                        for status in RECOVERY_STATUSES
                        if status_counts.get(status, 0)
                    ]
                )
                chart = (
                    alt.Chart(status_data)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("Status:N", sort=None, title=None),
                        y=alt.Y("Records:Q", title="Recovery records"),
                        color=alt.Color(
                            "Status:N",
                            scale=alt.Scale(
                                domain=list(status_colors),
                                range=list(status_colors.values()),
                            ),
                            legend=None,
                        ),
                        tooltip=["Status:N", "Records:Q"],
                    )
                )
                st.altair_chart(
                    chart,
                    alt="Relationship recovery records by current status",
                )
            else:
                st.info("No recovery records are available.")
    with timeline_column:
        with st.container(border=True):
            st.subheader("Memory creation timeline")
            if snapshot["timeline"]:
                import pandas as pd

                timeline = pd.DataFrame(snapshot["timeline"])
                st.bar_chart(
                    timeline,
                    x="date",
                    y="memories",
                    alt="Memories grouped by their recorded creation date",
                )
                if snapshot["nodes_missing_created_at_count"]:
                    st.caption(
                        f"Includes only memories with a saved creation date; "
                        f"{snapshot['nodes_missing_created_at_count']:,} "
                        "memories are omitted."
                    )
            else:
                if snapshot["node_count"] == 0:
                    st.info(
                        "No memories are stored, so there is no creation "
                        "activity to chart."
                    )
                else:
                    missing_dates = snapshot[
                        "nodes_missing_created_at_count"
                    ]
                    st.info(
                        f"A memory-creation timeline is unavailable for "
                        f"{missing_dates:,} stored memories because their "
                        "creation dates were not saved. Persist a creation "
                        "timestamp when each graph node is created to enable "
                        "this chart."
                    )

    recent = sorted(
        (
            memory for memory in snapshot["memory_items"]
            if memory["created_sort"] is not None
        ),
        key=lambda item: item["created_sort"],
        reverse=True,
    )[:8]
    with st.container(border=True):
        st.subheader("Recently created memories")
        if recent:
            st.dataframe(
                _memory_rows(recent),
                hide_index=True,
                alt="Most recently created memories with timestamps available",
            )
        else:
            if snapshot["node_count"] == 0:
                st.info("No memories have been created for this graph yet.")
            else:
                st.info(
                    "Recent memory activity cannot be ordered because no "
                    "valid creation dates are saved."
                )
    if snapshot["recovery_records"]:
        st.caption(
            "Recovery records do not store event timestamps, so a recent "
            "recovery activity feed is not available."
        )


def _render_memory_explorer(st, snapshot):
    st.header("Memory explorer", icon=":material/search:")
    memories = snapshot["memory_items"]
    status_options = sorted(
        {
            (memory.get("recovery") or {}).get("status", "untracked")
            for memory in memories
        }
    )
    filters = st.columns([2, 1, 1, 1])
    with filters[0]:
        query = st.text_input(
            "Search memory text or ID",
            type="search",
            placeholder="Search across the selected user's memories",
        )
    with filters[1]:
        selected_statuses = st.multiselect(
            "Recovery status",
            status_options,
            format_func=_status_label,
            help="Leave empty to include every recovery status.",
        )
    with filters[2]:
        connectivity = st.selectbox(
            "Connectivity",
            ("All", "connected", "isolated"),
        )
    with filters[3]:
        sort_by = st.selectbox(
            "Sort by",
            (
                "Most recently updated",
                "Most recently created",
                "Most connected",
                "Memory ID",
            ),
        )

    filtered = filter_memory_items(
        memories,
        query=query,
        recovery_statuses=selected_statuses,
        connectivity=connectivity,
        sort_by=sort_by,
    )
    page_size = st.selectbox(
        "Rows per page",
        (10, 25, 50, 100),
        index=1,
    )
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    current_page = min(
        int(st.session_state.get("dashboard_memory_page", 1)),
        total_pages,
    )
    st.session_state["dashboard_memory_page"] = current_page
    page_column, count_column = st.columns([1, 3])
    with page_column:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key="dashboard_memory_page",
        )
    with count_column:
        st.caption(
            f"{len(filtered):,} matching memories · page {page} of "
            f"{total_pages}"
        )
    page_items, _ = paginate(filtered, int(page), page_size)
    if page_items:
        st.dataframe(
            _memory_rows(page_items),
            hide_index=True,
            alt="Filtered memory records",
        )
    else:
        st.info("No memories match these search and filter settings.")

    include_text = st.checkbox(
        "Include redacted memory text in the CSV export",
        value=False,
        help="Memory text is excluded by default. Sensitive patterns are "
             "redacted when included.",
    )
    st.download_button(
        "Download filtered results",
        data=export_memories_csv(filtered, include_text),
        file_name="memories.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("Memory inspector")
    if not memories:
        st.info("There are no memories to inspect.")
        return
    options = [memory["id"] for memory in memories]
    display_by_id = {
        memory["id"]: memory["display_id"] for memory in memories
    }
    graph_label_by_id = {
        memory["id"]: memory["graph_label"] for memory in memories
    }
    default = (
        page_items[0]["id"] if page_items else options[0]
    )
    selected_id = st.selectbox(
        "Select a memory",
        options,
        index=options.index(default),
        format_func=lambda value: graph_label_by_id[value],
        key="dashboard_inspected_memory",
    )
    selected = next(
        memory for memory in memories if memory["id"] == selected_id
    )
    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("**Memory text**")
            st.text_area(
                "Redacted memory text",
                value=selected["memory_text"] or "No text available",
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )
    with right:
        with st.container(border=True):
            st.markdown("**Memory details**")
            st.write(f"Memory ID: `{selected['display_id']}`")
            st.write(f"Created: {_format_timestamp(selected['created_at'])}")
            st.write(f"Updated: {_format_timestamp(selected['updated_at'])}")
            st.write(f"Connectivity: {selected['connectivity']}")
            st.write(f"Relationships: {selected['relationship_count']}")
            recovery = selected.get("recovery")
            if recovery:
                st.write(f"Recovery: {_status_label(recovery['status'])}")
                st.write(
                    f"Attempts: {recovery['attempts']} / "
                    f"{recovery['attempt_cap']}"
                )
                st.write(
                    f"Eligible: {'Yes' if recovery['eligible'] else 'No'} "
                    f"({recovery['eligibility_reason']})"
                )
                if recovery["replacement_id"]:
                    st.write(
                        f"Replacement: `{recovery['replacement_id']}`"
                    )
                if recovery["error"]:
                    st.caption(f"Last error: {recovery['error']}")
            else:
                st.caption("No valid recovery record is associated with this node.")
            if st.button(
                "Open in knowledge graph",
                key="open_memory_in_graph",
                icon=":material/hub:",
            ):
                st.session_state["dashboard_graph_selected_node"] = selected_id
                st.session_state["dashboard_section"] = "Knowledge graph"
                st.rerun()
    related = _connected_memories(snapshot, selected_id)
    st.markdown("**Connected memories**")
    if related:
        st.dataframe(
            [
                {
                    "Memory": item["preview"],
                    "Relationship": (
                        f"{item['direction']} · {item['relationship']}"
                    ),
                    "Memory ID": item["memory_id_display"],
                    "Endpoint": (
                        "Missing" if item["missing_endpoint"] else "Present"
                    ),
                }
                for item in related
            ],
            hide_index=True,
            alt="Memories directly connected to the selected node",
        )
        jump_options = [
            item["memory_id"]
            for item in related
            if item["memory_id"] is not None
        ]
        if jump_options:
            jump_to = st.selectbox(
                "Inspect a connected memory",
                jump_options,
                format_func=lambda value: (
                    f"{graph_label_by_id[value]} · "
                    f"ID …{display_by_id[value][-6:]}"
                ),
                key="dashboard_connected_memory",
            )
            connected_memory = next(
                memory for memory in memories if memory["id"] == jump_to
            )
            st.markdown("**Selected connected memory**")
            st.write(connected_memory["memory_text"] or "Memory text unavailable")
            st.caption(f"Memory ID: {connected_memory['display_id']}")
            if st.button(
                "Show connected memory in graph",
                key="show_connected_in_graph",
                icon=":material/hub:",
            ):
                st.session_state["dashboard_graph_selected_node"] = jump_to
                st.session_state["dashboard_section"] = "Knowledge graph"
                st.rerun()
    else:
        st.info("This memory has no valid connected nodes.")


def _render_graph(st, snapshot):
    st.header("Knowledge graph", icon=":material/hub:")
    memories = snapshot["memory_items"]
    relationships = snapshot["relationship_items"]
    if not memories:
        st.info("No graph nodes are available to visualize.")
        return

    search = st.text_input(
        "Search memories",
        type="search",
        placeholder="Search memory text or ID",
    ).casefold().strip()
    matching_ids = [
        memory["id"]
        for memory in memories
        if (
            search in memory["id"].casefold()
            or search in memory["memory_text"].casefold()
            or search in memory["graph_preview"].casefold()
        )
    ][:500]
    if not matching_ids:
        st.info("No memories match that search.")
        return
    display_by_id = {
        memory["id"]: memory["display_id"] for memory in memories
    }
    graph_label_by_id = {
        memory["id"]: memory["graph_label"] for memory in memories
    }
    selected_state_key = "dashboard_graph_selected_node"
    if st.session_state.get(selected_state_key) not in matching_ids:
        st.session_state[selected_state_key] = matching_ids[0]
    selected_id = st.selectbox(
        "Select a memory",
        matching_ids,
        format_func=lambda value: (
            f"{graph_label_by_id[value]} · ID …{display_by_id[value][-6:]}"
        ),
        key=selected_state_key,
    )
    relationship_types = st.multiselect(
        "Relationship types",
        snapshot["relationship_types"],
        help="Leave empty to include all relationship types.",
    )
    node_limit = st.slider(
        "Maximum nodes to render",
        min_value=1,
        max_value=min(MAX_GRAPH_NODES, max(1, len(memories))),
        value=min(50, max(1, len(memories))),
        step=1,
    )
    selected_nodes, selected_edges = graph_subset(
        memories,
        relationships,
        relationship_types=relationship_types,
        selected_node_id=selected_id,
        max_nodes=node_limit,
    )
    if len(memories) > len(selected_nodes):
        st.caption(
            f"Rendering {len(selected_nodes)} of {len(memories):,} nodes. "
            "The selected node and its available neighbors are prioritized."
        )
    st.mermaid_chart(
        build_graph_mermaid(selected_nodes, selected_edges, selected_id),
        width="stretch",
        alt="Selected memory graph with relationships and isolated nodes",
    )
    selected_memory = next(
        memory for memory in memories if memory["id"] == selected_id
    )
    st.subheader("Selected memory")
    details, text = st.columns([1, 2])
    with details:
        st.write(f"Memory ID: `{selected_memory['display_id']}`")
        st.write(f"Connectivity: {selected_memory['connectivity']}")
        st.write(f"Relationship count: {selected_memory['relationship_count']}")
        st.write(f"Created: {_format_timestamp(selected_memory['created_at'])}")
        st.write(
            f"Last updated: {_format_timestamp(selected_memory['updated_at'])}"
        )
    with text:
        st.text_area(
            "Full memory text",
            value=(
                selected_memory["memory_text"]
                or "Memory text unavailable"
            ),
            height=140,
            disabled=True,
        )
    related = _connected_memories(snapshot, selected_id)
    st.markdown("**Connected memories**")
    if related:
        st.dataframe(
            [
                {
                    "Memory": item["preview"],
                    "Relationship": (
                        f"{item['direction']} · {item['relationship']}"
                    ),
                    "Memory ID": item["memory_id_display"],
                    "Endpoint": (
                        "Missing" if item["missing_endpoint"] else "Present"
                    ),
                }
                for item in related
            ],
            hide_index=True,
            alt="Connected memories with relationship directions and endpoint status",
        )
        connected_options = [
            item["memory_id"]
            for item in related
            if item["memory_id"] is not None
        ]
        if connected_options:
            connected_state_key = "dashboard_graph_connected_node"
            if st.session_state.get(connected_state_key) not in connected_options:
                st.session_state[connected_state_key] = None
            st.selectbox(
                "Navigate to a connected memory",
                connected_options,
                index=None,
                placeholder="Choose a connected memory",
                format_func=lambda value: (
                    f"{graph_label_by_id[value]} · "
                    f"ID …{display_by_id[value][-6:]}"
                ),
                key=connected_state_key,
                on_change=_select_connected_graph_memory,
            )
    else:
        st.info("This memory has no relationship records.")
    missing_edges = [
        edge
        for edge in relationships
        if edge["missing_endpoint"]
        and selected_id in (edge["source"], edge["target"])
    ]
    if missing_edges:
        st.warning(
            "One or more relationships for this memory point to a missing "
            "memory. These endpoints are shown as missing above and are not "
            "drawn as valid graph connections."
        )
    elif any(edge["missing_endpoint"] for edge in relationships):
        st.caption(
            "Other graph relationships reference missing endpoints; they "
            "are not drawn as valid connections."
        )


def _render_recovery_centre(st, snapshot):
    st.header("Recovery centre", icon=":material/build:")
    st.caption(
        "Read-only view. Recovery generation, enrollment, restoration, and "
        "retries remain available only through the existing CLI."
    )
    with st.container(horizontal=True):
        st.metric("Eligible", snapshot["eligible_recovery_count"], border=True)
        st.metric("At attempt cap", snapshot["capped_recovery_count"], border=True)
        st.metric(
            "Invalid records",
            snapshot["recovery_record_count"]
            - snapshot["valid_recovery_record_count"],
            border=True,
        )
    attention_statuses = {
        "failed",
        "blocked_missing_endpoint",
        "pending_persistence",
        "invalid",
    }
    attention = [
        record
        for record in snapshot["recovery_records"]
        if (
            record["status"] in attention_statuses
            or record["eligibility_reason"] == "attempt_cap_reached"
        )
    ]
    if attention:
        st.warning(
            f"{len(attention)} unresolved recovery record(s) may need "
            "operator attention.",
            icon=":material/warning:",
        )
        memories_by_id = {
            item["id"]: item for item in snapshot["memory_items"]
        }
        st.dataframe(
            [
                {
                    "Memory": (
                        memories_by_id.get(
                            record.get("source_id"),
                            {},
                        ).get("preview", "Memory text unavailable")
                    ),
                    "Why it matters": _recovery_attention_action(record),
                    "Next step": (
                        "Inspect this read-only record, then use the CLI if "
                        "an action is appropriate."
                    ),
                }
                for record in attention
            ],
            hide_index=True,
            alt="Recovery records that may need operator attention and suggested checks",
        )
    else:
        st.success("No failed, blocked, or capped recovery records.")
    rows = _recovery_rows(snapshot)
    if rows:
        st.dataframe(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "Memory ID",
                        "Technical status",
                        "Replacement ID",
                    }
                }
                for row in rows
            ],
            hide_index=True,
            alt="Validated relationship recovery records and eligibility",
        )
        with st.expander("Technical recovery details"):
            st.dataframe(
                [
                    {
                        "Memory ID": row["Memory ID"],
                        "Stored status": row["Technical status"],
                        "Replacement ID": row["Replacement ID"],
                    }
                    for row in rows
                ],
                hide_index=True,
                alt="Stored recovery statuses and IDs for debugging",
            )
        historical_count = snapshot["superseded_recovery_count"]
        if historical_count:
            st.caption(
                f"{historical_count:,} superseded record(s) are historical "
                "replacement links and are not treated as unresolved work."
            )
    else:
        st.info("No relationship recovery records are present.")


def _render_integrity(st, snapshot):
    st.header("Data integrity", icon=":material/verified:")
    integrity = snapshot["integrity"]
    error_column, warning_column = st.columns(2)
    error_column.metric("Errors", integrity["error_count"])
    warning_column.metric("Warnings", integrity["warning_count"])
    if not integrity["issues"]:
        st.success("No integrity issues were detected.")
        return
    st.dataframe(
        _integrity_rows(integrity["issues"]),
        hide_index=True,
        alt="Graph integrity findings with severity, impact, and recommended next steps",
    )
    st.caption(
        "Recovery transition history is not part of the persisted schema, "
        "so historical transition validity cannot be verified."
    )


def _render_diagnostics(st, snapshot):
    st.header("Performance and diagnostics", icon=":material/query_stats:")
    st.caption(
        "Graph summaries are computed from the loaded local snapshot. No "
        "Mem0 or Gemini calls are made by this dashboard."
    )
    graph_path = snapshot["graph_path"] or "Unavailable"
    st.write(f"Data provider: JSON graph snapshot ({graph_path})")
    st.write(
        "Last graph-file modification: "
        f"{_format_timestamp(snapshot['graph_modified_at'])}"
    )
    st.write(
        f"File freshness: {graph_freshness_label(snapshot['graph_modified_at'])}"
    )
    st.write(
        f"Graph size: {snapshot['node_count']:,} nodes · "
        f"{snapshot['relationship_count']:,} relationships · "
        f"{snapshot['recovery_record_count']:,} recovery records"
    )
    st.info(
        "Operation timings are printed by the CLI performance instrumentation "
        "but are not persisted. Historical latency charts and error trends "
        "cannot be produced yet. Future reporting needs a bounded, "
        "user-scoped operation-event store with operation category, duration, "
        "success/failure, and timestamp—never prompts, memory text, IDs, or "
        "credentials.",
        title="No historical performance store",
    )


def main():
    import streamlit as st

    st.set_page_config(
        page_title="Mem0 memory operations dashboard",
        page_icon=":material/hub:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    @st.cache_data(ttl=30, max_entries=16, show_spinner=False)
    def cached_snapshot(graph_path, modified_ns, file_size, user_id):
        del modified_ns, file_size
        return load_dashboard_snapshot(graph_path, user_id)

    with st.sidebar:
        st.title("Memory operations dashboard")
        st.caption("Read-only graph and recovery observability")
        with st.form("dashboard_scope"):
            user_id = st.text_input(
                "Local user ID",
                value=st.session_state.get("dashboard_user_id", ""),
                help="Selects a user-scoped graph snapshot; it is not "
                     "an authentication mechanism.",
            )
            graph_path = st.text_input(
                "Graph file",
                value=st.session_state.get("dashboard_graph_path", GRAPH_FILE),
            )
            refresh = st.form_submit_button(
                "Load or refresh snapshot",
                type="primary",
                icon=":material/refresh:",
                width="stretch",
            )
        if refresh:
            st.session_state["dashboard_user_id"] = user_id.strip()
            st.session_state["dashboard_graph_path"] = graph_path.strip()
            cached_snapshot.clear()
        st.warning(
            "Local operator view only. A user ID selects data but does not "
            "prove authorization. Do not expose this app publicly until "
            "authentication and tenant authorization are implemented.",
            icon=":material/lock:",
        )
        st.caption(
            "The dashboard never writes graph/session files or runs recovery."
        )

    active_user = st.session_state.get("dashboard_user_id", "")
    active_graph_path = st.session_state.get(
        "dashboard_graph_path",
        GRAPH_FILE,
    )
    st.title("Mem0 memory operations")
    st.caption(
        "A read-only view of graph health, memory connectivity, and "
        "relationship recovery."
    )
    if not active_user:
        st.info(
            "Choose a local user ID and graph file in the sidebar, then load "
            "the snapshot.",
            title="Select a data scope",
        )
        return
    if not active_graph_path:
        st.error("Enter a graph file path.", title="Graph source unavailable")
        return

    modified_ns, file_size = _stat_key(active_graph_path)
    try:
        with st.spinner("Loading the selected graph snapshot…"):
            snapshot = cached_snapshot(
                str(Path(active_graph_path).expanduser().resolve()),
                modified_ns,
                file_size,
                active_user,
            )
    except DashboardDataError as error:
        st.error(str(error), title="Could not load graph snapshot")
        return

    st.caption(
        f"Scope: selected local account · provider: JSON snapshot · "
        f"{graph_freshness_label(snapshot['graph_modified_at'])}"
    )
    if snapshot["integrity"]["error_count"]:
        st.warning(
            f"{snapshot['integrity']['error_count']} integrity error(s) "
            "were found. Review the Data integrity section.",
            icon=":material/warning:",
        )

    section = st.segmented_control(
        "Dashboard section",
        (
            "Overview",
            "Memory explorer",
            "Knowledge graph",
            "Recovery centre",
            "Data integrity",
            "Performance and diagnostics",
        ),
        default="Overview",
        key="dashboard_section",
    )
    if section == "Memory explorer":
        _render_memory_explorer(st, snapshot)
    elif section == "Knowledge graph":
        _render_graph(st, snapshot)
    elif section == "Recovery centre":
        _render_recovery_centre(st, snapshot)
    elif section == "Data integrity":
        _render_integrity(st, snapshot)
    elif section == "Performance and diagnostics":
        _render_diagnostics(st, snapshot)
    else:
        _render_overview(st, snapshot)


if __name__ == "__main__":
    main()
