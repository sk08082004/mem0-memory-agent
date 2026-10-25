import json


COMMANDS = (
    ("/memories", "List current saved memories."),
    ("/memory-info <number>", "Show provenance and update history for a memory."),
    ("/graph", "Show the knowledge graph, memories, and relationships."),
    ("/related <memory_number>", "List memories connected to a graph memory."),
    (
        "/recover-relationships",
        "Preview relationship recovery without changing data.",
    ),
    (
        "/recover-relationships --apply",
        "Apply the next eligible relationship-recovery batch.",
    ),
    (
        "/recover-relationships --enroll <memory_id>",
        "Track one graph node for recovery without running generation.",
    ),
    (
        "/recover-relationships --apply-source <memory_id>",
        "Apply relationship recovery for one source.",
    ),
    (
        "/recover-relationships --inspect-source <memory_id>",
        "Inspect one recovery source without changing data.",
    ),
    (
        "/recover-relationships --verify-remote-source <memory_id>",
        "Verify one memory ID in the remote Mem0 listing.",
    ),
    (
        "/recover-relationships --restore-local-source <memory_id>",
        "Restore a verified local recovery source.",
    ),
    (
        "/reconcile-memory-events [limit]",
        "Check pending Mem0 events and synchronize confirmed graph nodes.",
    ),
    ("/forget <number>", "Delete one current memory by its list number."),
    ("/clear", "Clear all memories after explicit confirmation."),
    ("/new", "Start a fresh conversation while keeping saved memories."),
    ("/user", "Show the currently selected user."),
    ("/login <user_id>", "Switch to another user."),
    ("/logout", "Log out and choose another user."),
    ("/help", "Show this list of commands and usage."),
    ("exit", "Exit the command-line agent."),
)


def format_command_help():
    width = max(len(syntax) for syntax, _ in COMMANDS)
    return "Available commands:\n\n" + "\n".join(
        f"{syntax:<{width}}  {description}"
        for syntax, description in COMMANDS
    )


def command_requires_agent(text):
    if not isinstance(text, str):
        return False
    command = text.strip().split(maxsplit=1)[0].casefold() if text.strip() else ""
    return command in {
        "/memories",
        "/graph",
        "/related",
        "/memory-info",
        "/forget",
        "/clear",
        "/recover-relationships",
        "/reconcile-memory-events",
    } or command.startswith(("/related", "/memory-info", "/forget"))


def dispatch_chat_command(agent, text, *, confirm_clear=False, user_id=None):
    """Run one recognized CLI command against the same Agent operations."""
    if not isinstance(text, str):
        return None
    command_text = text.strip()
    if not command_text:
        return None
    parts = command_text.split()
    command = parts[0].casefold()

    if command_text.casefold() == "exit":
        return _result(
            "exit",
            "In the CLI, `exit` ends the agent session. The dashboard stays open.",
        )
    if command == "/help" and len(parts) == 1:
        return _result("help", format_command_help(), "markdown")
    if command == "/memories" and len(parts) == 1:
        try:
            from app.main import get_active_memories

            memories = get_active_memories(agent)
            if not memories:
                return _result("memories", "No memories stored.")
            rows = [
                f"{index}. {memory.get('memory', '')}"
                for index, memory in enumerate(memories, 1)
            ]
            return _result("memories", "Stored memories:\n\n" + "\n".join(rows))
        except Exception as error:
            return _error("memories", f"Could not retrieve memories: {error}")
    if command == "/graph" and len(parts) == 1:
        try:
            memories = agent.memory.get_all_complete(agent.user_id)
            agent.graph.sync_user_graph(agent.user_id, memories)
            graph = agent.graph.get_user_graph(agent.user_id)
            nodes = graph.get("nodes", [])
            relationships = graph.get("relationships", [])
            lines = [
                "## Knowledge graph",
                f"Nodes: {len(nodes)}",
                f"Relationships: {len(relationships)}",
            ]
            if nodes:
                lines.extend(["", "### Memories"])
                node_numbers = {}
                for index, node in enumerate(nodes, 1):
                    node_numbers[node["id"]] = index
                    lines.append(f"{index}. {node['memory_text']}")
                if relationships:
                    lines.extend(["", "### Relationships"])
                    lines.extend(
                        f"{node_numbers.get(edge['source'], '?')} "
                        f"--{edge['relationship']}--> "
                        f"{node_numbers.get(edge['target'], '?')}"
                        for edge in relationships
                    )
                else:
                    lines.extend(["", "No relationships in the graph."])
            else:
                lines.extend(["", "No nodes in the graph."])
            return _result("graph", "\n".join(lines), "markdown")
        except Exception as error:
            return _error("graph", f"Could not retrieve knowledge graph: {error}")
    if command == "/related" or command.startswith("/related"):
        return _related(agent, parts)
    if command == "/memory-info" or command.startswith("/memory-info"):
        return _memory_info(agent, parts)
    if command == "/user" and len(parts) == 1:
        current_user = user_id or getattr(agent, "user_id", "")
        return _result("user", f"Current user: `{current_user}`")
    if command == "/login" or command.startswith("/login"):
        if len(parts) != 2:
            return _error("login", "Usage: `/login <user_id>`")
        target_user = parts[1].strip()
        if not target_user:
            return _error("login", "User ID cannot be empty.")
        return _result(
            "login",
            f"Switching the dashboard to user `{target_user}`.",
            action={"type": "login", "user_id": target_user},
        )
    if command == "/logout" and len(parts) == 1:
        return _result(
            "logout",
            "Logged out of the current dashboard scope. Select another local "
            "user in Workspace settings or use `/login <user_id>`.",
            action={"type": "logout"},
        )
    if command == "/forget" or command.startswith("/forget"):
        return _forget(agent, parts)
    if command == "/clear" and len(parts) == 1:
        return _clear(agent, confirm_clear)
    if command == "/new" and len(parts) == 1:
        return _result(
            "new",
            "Starting a new conversation. Saved memories are unchanged.",
            action={"type": "new_conversation"},
        )
    if command == "/recover-relationships":
        return _recover_relationships(agent, parts)
    if command == "/reconcile-memory-events":
        return _reconcile_memory_events(agent, parts)
    if command.startswith("/"):
        return _error(
            "unknown",
            f"Unknown command `{parts[0]}`. Type `/help` to see supported commands.",
        )
    return None


def _result(command, text, response_format="text", action=None):
    return {
        "handled": True,
        "command": command,
        "status": "success",
        "text": text,
        "format": response_format,
        "action": action,
    }


def _error(command, text):
    return {
        "handled": True,
        "command": command,
        "status": "error",
        "text": text,
        "format": "text",
        "action": None,
    }


def _memory_number(parts, command, usage, invalid_message):
    if len(parts) != 2:
        return None, _error(command, f"Usage: `{usage}`")
    try:
        number = int(parts[1])
    except ValueError:
        return None, _error(command, invalid_message)
    return number, None


def _related(agent, parts):
    number, error = _memory_number(
        parts,
        "related",
        "/related <memory_number>",
        "Please enter a valid memory number.",
    )
    if error:
        return error
    try:
        memories = agent.memory.get_all_complete(agent.user_id)
        agent.graph.sync_user_graph(agent.user_id, memories)
        graph = agent.graph.get_user_graph(agent.user_id)
        nodes = graph.get("nodes", [])
        relationships = graph.get("relationships", [])
        if number < 1 or number > len(nodes):
            return _error("related", "Invalid memory number.")
        selected = nodes[number - 1]
        related = []
        for edge in relationships:
            if edge["source"] == selected["id"]:
                related_id = edge["target"]
            elif edge["target"] == selected["id"]:
                related_id = edge["source"]
            else:
                continue
            node = next((item for item in nodes if item["id"] == related_id), None)
            if node:
                related.append(
                    f"- {node['memory_text']}\n"
                    f"  Relationship: {edge['relationship']}"
                )
        content = (
            f"Memories related to: {selected['memory_text']}\n\n"
            "Related memories:\n\n"
            + ("\n".join(related) if related else "No related memories found.")
        )
        return _result("related", content, "markdown")
    except Exception as error:
        return _error("related", f"Could not retrieve related memories: {error}")


def _memory_info(agent, parts):
    number, error = _memory_number(
        parts,
        "memory-info",
        "/memory-info <number>",
        "Please enter a valid memory number.",
    )
    if error:
        return error
    try:
        from app.main import get_active_memories

        memories = get_active_memories(agent)
        if number < 1 or number > len(memories):
            return _error("memory-info", "Invalid memory number.")
        memory = memories[number - 1]
        metadata = memory.get("metadata", {})
        details = [
            "## Memory details",
            f"**Memory:** {memory.get('memory', '')}",
            f"**Importance:** {metadata.get('importance', 'N/A')}",
            f"**Confidence:** {metadata.get('confidence', 'N/A')}",
            f"**Reason:** {metadata.get('reason', 'N/A')}",
            f"**Source:** {metadata.get('source', 'N/A')}",
            f"**Created:** {memory.get('created_at', 'N/A')}",
            f"**Updated:** {memory.get('updated_at', 'N/A')}",
        ]
        evolution = metadata.get("evolution")
        if isinstance(evolution, str):
            try:
                evolution = json.loads(evolution)
            except (json.JSONDecodeError, TypeError):
                evolution = None
        history = evolution.get("history", []) if isinstance(evolution, dict) else []
        if history:
            details.extend(["", "### Timeline", f"Originally: {history[0].get('from', 'N/A')}"])
            details.extend(
                f"Updated to: {change.get('to', 'N/A')}"
                for change in history
                if isinstance(change, dict)
            )
        return _result("memory-info", "\n".join(details), "markdown")
    except Exception as error:
        return _error("memory-info", f"Could not retrieve memory details: {error}")


def _forget(agent, parts):
    number, error = _memory_number(
        parts,
        "forget",
        "/forget <number>",
        "Please enter a memory number.",
    )
    if error:
        return error
    try:
        from app.main import get_active_memories

        memories = get_active_memories(agent)
        if number < 1 or number > len(memories):
            return _error("forget", "Invalid memory number.")
        memory = memories[number - 1]
        memory_id = memory["id"]
        deletion = agent.memory.delete(memory_id)
        status = agent.memory.delete_result_status(deletion)
        if status == "confirmed":
            try:
                agent.graph.delete_node(agent.user_id, memory_id)
            except Exception as error:
                return _result(
                    "forget",
                    "Memory deletion was confirmed, but graph cleanup failed: "
                    f"{error}",
                )
            return _result("forget", f"Memory deleted: {memory['memory']}")
        if status == "failed":
            return _error(
                "forget",
                "Mem0 reported that deletion failed; the graph was left unchanged.",
            )
        return _result(
            "forget",
            "Memory deletion could not be confirmed; the graph was left unchanged.",
        )
    except Exception as error:
        return _error("forget", f"Could not delete memory: {error}")


def _clear(agent, confirmed):
    try:
        memories = agent.memory.get_all_complete(agent.user_id)
        if memories["count"] == 0:
            agent.graph.sync_user_graph(agent.user_id, memories)
            return _result("clear", "No memories to clear.")
        if not confirmed:
            return {
                "handled": True,
                "command": "clear",
                "status": "confirmation",
                "text": (
                    f"This will delete {memories['count']} memories. "
                    "Confirm to continue."
                ),
                "format": "text",
                "action": {"type": "confirm_clear"},
            }
        result = agent.memory.clear(agent.user_id)
        if not result["retrieval_complete"]:
            return _error(
                "clear",
                f"Could not safely clear memories: {result['error']}",
            )
        graph_synced = False
        warning = None
        if result["remaining_retrieval_complete"]:
            try:
                agent.graph.sync_user_graph(
                    agent.user_id,
                    result["remaining_memories"],
                )
                graph_synced = True
            except Exception as error:
                warning = f"Graph synchronization failed: {error}"
        else:
            warning = (
                "Could not verify remaining memories; the graph was left "
                f"unchanged. Details: {result['error']}"
            )
        if result.get("complete") and graph_synced:
            return _result("clear", "All memories were cleared and the graph synchronized.")
        details = [
            "Clear was partial or could not be fully confirmed.",
            f"Confirmed deleted IDs: {result['deleted_ids']}",
            f"Failed IDs: {result['failed_ids']}",
            f"Unconfirmed IDs: {result['unconfirmed_ids']}",
            f"Remaining IDs: {result['remaining_ids']}",
        ]
        if warning:
            details.insert(1, warning)
        return _result("clear", "\n".join(details), "markdown")
    except Exception as error:
        return _error("clear", f"Could not retrieve memories: {error}")


def _recover_relationships(agent, parts):
    mode = "preview"
    source_id = None
    if len(parts) == 2 and parts[1].casefold() == "--apply":
        mode = "apply"
    elif len(parts) == 3 and parts[1].casefold() in {
        "--enroll",
        "--apply-source",
        "--inspect-source",
        "--verify-remote-source",
        "--restore-local-source",
    }:
        source_id = parts[2]
        mode = {
            "--enroll": "enroll",
            "--apply-source": "apply_source",
            "--inspect-source": "inspect_source",
            "--verify-remote-source": "verify_remote_source",
            "--restore-local-source": "restore_local_source",
        }[parts[1].casefold()]
    elif len(parts) != 1:
        return _error(
            "recover-relationships",
            "Usage: `/recover-relationships [--apply | --enroll <memory_id> | "
            "--apply-source <memory_id> | --inspect-source <memory_id> | "
            "--verify-remote-source <memory_id> | "
            "--restore-local-source <memory_id>]`",
        )
    try:
        if mode == "enroll":
            report = agent.enroll_relationship_recovery(source_id)
            status = report["status"]
            if status == "enrolled":
                text = "Relationship recovery source enrolled. No generation was run."
            elif status == "already_tracked":
                text = (
                    "Source already has a recovery record "
                    f"({report['record_status']}); it was not changed."
                )
            else:
                text = (
                    "Could not enroll recovery source: "
                    f"{status}"
                    + (f" ({report['reason']})" if report.get("reason") else "")
                )
            return _result("recover-relationships", text)
        if mode == "inspect_source":
            report = agent.inspect_relationship_recovery(source_id)
            return _result(
                "recover-relationships",
                json.dumps(report, indent=2, ensure_ascii=False),
                "code",
            )
        if mode == "verify_remote_source":
            report = agent.verify_remote_memory(source_id)
            return _result(
                "recover-relationships",
                json.dumps(report, indent=2, ensure_ascii=False),
                "code",
            )
        if mode == "restore_local_source":
            report = agent.restore_local_recovery_source(source_id)
            return _result(
                "recover-relationships",
                json.dumps(report, indent=2, ensure_ascii=False),
                "code",
            )
        apply_recovery = mode in {"apply", "apply_source"}
        report = agent.recover_relationships(
            dry_run=not apply_recovery,
            source_id=source_id if mode == "apply_source" else None,
        )
        if mode == "preview":
            try:
                report["runtime_diagnostics"] = agent.relationship_runtime_diagnostics(
                    eligibility_report=report,
                )
            except Exception as error:
                report["runtime_diagnostics_error"] = str(error)
        elif mode == "apply_source":
            try:
                report["runtime_diagnostics"] = agent.relationship_runtime_diagnostics(
                    source_id,
                )
            except Exception as error:
                report["runtime_diagnostics_error"] = str(error)
        return _result(
            "recover-relationships",
            json.dumps(report, indent=2, ensure_ascii=False),
            "code",
        )
    except Exception as error:
        return _error(
            "recover-relationships",
            f"Relationship recovery failed: {error}",
        )


def _reconcile_memory_events(agent, parts):
    if len(parts) > 2:
        return _error(
            "reconcile-memory-events",
            "Usage: `/reconcile-memory-events [limit]`",
        )
    limit = 10
    if len(parts) == 2:
        try:
            limit = int(parts[1])
        except ValueError:
            return _error(
                "reconcile-memory-events",
                "Limit must be a positive whole number.",
            )
        if limit < 1:
            return _error(
                "reconcile-memory-events",
                "Limit must be a positive whole number.",
            )
    try:
        report = agent.reconcile_pending_memory_events(limit=limit)
    except Exception as error:
        print(f"[ERROR] Mem0 event reconciliation failed: {error}")
        return _error(
            "reconcile-memory-events",
            "Could not reconcile pending Mem0 events. Check application logs "
            "for diagnostic details.",
        )
    details = [
        (
            f"- {item.get('operation_type', 'operation')}: "
            f"{item.get('status', 'unknown')}"
        )
        for item in report.get("details", [])
        if isinstance(item, dict)
    ]
    lines = [
        "## Mem0 event reconciliation",
        f"Checked: {report['checked']}",
        f"Graph synchronized: {report['synchronized']}",
        f"Still pending: {report['pending']}",
        f"Relationship recovery untracked: {report.get('untracked', 0)}",
        f"Failed: {report['failed']}",
    ]
    if details:
        lines.extend(["", *details])
    return _result(
        "reconcile-memory-events",
        "\n".join(lines),
        "markdown",
    )
