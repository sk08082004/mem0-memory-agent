from app.agent import Agent
from app.config import SESSION_FILE
from app.conversation import MEMORY_OUTCOME_LABELS
from app.performance import finish_timer, start_timer
from app.commands import dispatch_chat_command, format_command_help
import json
import os
import tempfile


def normalize_user_id(user_id):
    if not isinstance(user_id, str):
        return None

    normalized_user_id = user_id.strip()
    return normalized_user_id or None


def create_agent(user_id):
    normalized_user_id = normalize_user_id(user_id)
    if normalized_user_id is None:
        raise ValueError("User ID must be a non-empty string.")

    agent = Agent(normalized_user_id)
    if getattr(agent, "user_id", None) != normalized_user_id:
        raise ValueError("Agent was initialized for an unexpected user ID.")

    return agent


def load_user():
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return None

        return normalize_user_id(data.get("user_id"))

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError
    ) as e:
        if isinstance(e, OSError) and not isinstance(e, FileNotFoundError):
            print(f"[WARNING] Could not read session file: {e}")
        return None


def save_user(user_id):
    normalized_user_id = normalize_user_id(user_id)
    if normalized_user_id is None:
        raise ValueError("User ID must be a non-empty string.")

    session_path = os.path.abspath(SESSION_FILE)
    session_directory = os.path.dirname(session_path)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=session_directory,
            delete=False
        ) as file:
            temporary_path = file.name
            json.dump({"user_id": normalized_user_id}, file)

        os.replace(temporary_path, session_path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def remove_user():
    try:
        os.remove(SESSION_FILE)
    except FileNotFoundError:
        pass


def print_recovery_runtime_diagnostics(
    agent,
    source_id=None,
    eligibility_report=None,
):
    diagnostics = agent.relationship_runtime_diagnostics(
        source_id,
        eligibility_report,
    )
    disk_fingerprint = (
        diagnostics["disk_fingerprint"]
        if diagnostics["disk_fingerprint"] is not None
        else f"unavailable ({diagnostics['disk_error'] or 'invalid_snapshot'})"
    )
    print(
        "[RECOVERY-STATE] "
        f"pid={diagnostics['process_id']} "
        f"graph={diagnostics['graph_path']}"
    )
    print(
        "[RECOVERY-STATE] "
        f"in_memory_sha256={diagnostics['memory_fingerprint'] or 'unavailable'} "
        f"on_disk_sha256={disk_fingerprint}"
    )
    print(
        "[RECOVERY-STATE] "
        f"memory_counts(nodes={diagnostics['node_count']}, "
        f"relationships={diagnostics['relationship_count']}, "
        f"records={diagnostics['recovery_record_count']}, "
        f"valid_records={diagnostics['valid_recovery_record_count']}) "
        f"disk_counts(nodes={diagnostics['disk_node_count']}, "
        f"relationships={diagnostics['disk_relationship_count']}, "
        f"records={diagnostics['disk_recovery_record_count']}, "
        f"valid_records={diagnostics['disk_valid_recovery_record_count']})"
    )
    if diagnostics["disk_error"]:
        print(
            f"[RECOVERY-STATE] on_disk_error={diagnostics['disk_error']}"
        )
    if source_id is not None:
        print(
            "[RECOVERY-STATE] "
            f"source={source_id} "
            f"node_exists={diagnostics['source_node_exists']} "
            f"valid_record={diagnostics['source_record_valid']} "
            f"status={diagnostics['source_record_status'] or 'unavailable'} "
            f"attempts={diagnostics['source_record_attempts'] if diagnostics['source_record_attempts'] is not None else 'unavailable'} "
            f"eligible={diagnostics['source_eligible']} "
            f"reason={diagnostics['source_eligibility_reason']}"
        )
    return diagnostics


def get_active_memories(agent):
    """Return only current memories, hiding historical timeline versions."""
    memories = agent.memory.get_all(agent.user_id)
    results = memories.get("results", [])

    historical_ids = set()

    for memory in results:
        evolution = memory.get("metadata", {}).get("evolution")

        if isinstance(evolution, str):
            try:
                evolution = json.loads(evolution)
            except (json.JSONDecodeError, TypeError):
                evolution = None

        if isinstance(evolution, dict):
            previous_id = evolution.get("previous_memory_id")
            if previous_id:
                historical_ids.add(previous_id)

            history = evolution.get("history", [])
            if isinstance(history, list):
                for change in history:
                    if isinstance(change, dict):
                        memory_id = change.get("memory_id")
                        if memory_id:
                            historical_ids.add(memory_id)

        elif isinstance(evolution, list):
            for item in evolution:
                if isinstance(item, str) and item.startswith("previous_memory_id."):
                    historical_ids.add(item.split(".", 1)[1])

    active_results = [
        memory
        for memory in results
        if memory.get("id") not in historical_ids
    ]

    return active_results


def main():
    print("*" * 50)
    print("MEM0 LONG-TERM MEMORY AGENT")
    print("=" * 50)
    print("Type 'exit' to quit.")
    print("Type '/help' to see your commands.\n")

    user_id = load_user()

    if user_id is None:
        while user_id is None:
            user_id = normalize_user_id(input("Enter your user ID: "))
            if user_id is None:
                print("\nUser ID cannot be empty.\n")

        agent = create_agent(user_id)
        save_user(user_id)
    else:
        agent = create_agent(user_id)

    while True:
        user_input = input("You: ").strip()

        # Exit command
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        command_parts = user_input.split(maxsplit=1)
        if (
            command_parts
            and command_parts[0].casefold() == "/reconcile-memory-events"
        ):
            command_start = start_timer()
            result = dispatch_chat_command(
                agent,
                user_input,
                user_id=user_id,
            )
            print(f"\n{result['text']}\n")
            finish_timer(
                "Command /reconcile-memory-events",
                command_start,
                result["status"] == "success",
            )
            continue

        # Show memories command
        if user_input.lower() == "/memories":
            command_start = start_timer()
            command_succeeded = True
            try:
                active_memories = get_active_memories(agent)

                print("\nStored memories:")

                if not active_memories:
                    print("No memories stored.")
                else:
                    for i, memory in enumerate(
                        active_memories,
                        start=1
                    ):
                        print(f"{i}. {memory['memory']}")

                print()

            except Exception as e:
                command_succeeded = False
                print(
                    f"\n[ERROR] Could not retrieve memories: {e}\n"
                )

            finish_timer(
                "Command /memories",
                command_start,
                command_succeeded,
            )
            continue

        recovery_parts = user_input.split()
        if recovery_parts and recovery_parts[0].lower() == "/recover-relationships":
            command_start = start_timer()
            command_succeeded = True
            recovery_mode = "invalid"
            source_id = None
            if len(recovery_parts) == 1:
                recovery_mode = "preview"
            elif len(recovery_parts) == 2 and recovery_parts[1].lower() == "--apply":
                recovery_mode = "apply"
            elif (
                len(recovery_parts) == 3
                and recovery_parts[1].lower() == "--enroll"
            ):
                recovery_mode = "enroll"
                source_id = recovery_parts[2]
            elif (
                len(recovery_parts) == 3
                and recovery_parts[1].lower() == "--apply-source"
            ):
                recovery_mode = "apply_source"
                source_id = recovery_parts[2]
            elif (
                len(recovery_parts) == 3
                and recovery_parts[1].lower() == "--inspect-source"
            ):
                recovery_mode = "inspect_source"
                source_id = recovery_parts[2]
            elif (
                len(recovery_parts) == 3
                and recovery_parts[1].lower() == "--verify-remote-source"
            ):
                recovery_mode = "verify_remote_source"
                source_id = recovery_parts[2]
            elif (
                len(recovery_parts) == 3
                and recovery_parts[1].lower() == "--restore-local-source"
            ):
                recovery_mode = "restore_local_source"
                source_id = recovery_parts[2]

            try:
                if recovery_mode == "invalid":
                    print(
                        "\nUsage: /recover-relationships "
                        "[--apply | --enroll <memory_id> | "
                        "--apply-source <memory_id> | "
                        "--inspect-source <memory_id> | "
                        "--verify-remote-source <memory_id> | "
                        "--restore-local-source <memory_id>]\n"
                    )
                    finish_timer(
                        "Command /recover-relationships",
                        command_start,
                        command_succeeded,
                    )
                    continue

                if recovery_mode == "enroll":
                    result = agent.enroll_relationship_recovery(source_id)
                    if result["status"] == "enrolled":
                        print(
                            "\nRelationship recovery source enrolled. "
                            "No generation was run.\n"
                        )
                    elif result["status"] == "already_tracked":
                        print(
                            "\nSource already has a recovery record "
                            f"({result['record_status']}); it was not changed.\n"
                        )
                    else:
                        print(
                            "\nCould not enroll recovery source: "
                            f"{result['status']}"
                            + (
                                f" ({result['reason']})"
                                if result.get("reason")
                                else ""
                            )
                            + "\n"
                        )
                    finish_timer(
                        "Command /recover-relationships",
                        command_start,
                        command_succeeded,
                    )
                    continue

                if recovery_mode == "inspect_source":
                    report = agent.inspect_relationship_recovery(source_id)
                    print(f"\nRelationship recovery inspection for {source_id}:")
                    print(f"Source node exists: {report['node_exists']}")
                    print(f"Valid recovery record exists: {report['record_valid']}")
                    if report["invalid_record"]:
                        print("Recovery record is invalid.")
                    elif report["status"] is not None:
                        print(f"Status: {report['status']}")
                        print(f"Attempts: {report['attempts']}")
                    print(f"Attempt cap: {report['attempt_cap']}")
                    print(
                        "Attempt cap reached: "
                        f"{report['attempt_cap_reached']}"
                    )
                    print(f"Eligible for recovery: {report['eligible']}")
                    print(f"Inspection result: {report['reason']}")
                    print(f"Active graph node count: {report['node_count']}")
                    print(
                        "Recovery record count: "
                        f"{report['recovery_record_count']}"
                    )
                    print(
                        "Valid recovery record count: "
                        f"{report['valid_recovery_record_count']}"
                    )
                    print(f"Recovery queue valid: {report['queue_valid']}")
                    print("Read-only inspection; no changes were made.\n")
                    print_recovery_runtime_diagnostics(agent, source_id)
                    finish_timer(
                        "Command /recover-relationships",
                        command_start,
                        command_succeeded,
                    )
                    continue

                if recovery_mode == "verify_remote_source":
                    report = agent.verify_remote_memory(source_id)
                    print(
                        "\nRemote Mem0 source verification "
                        f"for {source_id}:"
                    )
                    print(f"Listing complete: {report['listing_complete']}")
                    print(f"Listing validated: {report['listing_valid']}")
                    print(
                        "Exact memory ID found: "
                        + (
                            str(report["found"])
                            if report["found"] is not None
                            else "inconclusive"
                        )
                    )
                    print(f"Result: {report['result']}")
                    print(f"Reported memory count: {report['memory_count']}")
                    print(f"Retrieved result count: {report['result_count']}")
                    print("Local graph and recovery state were not changed.\n")
                    finish_timer(
                        "Command /recover-relationships",
                        command_start,
                        command_succeeded,
                    )
                    continue

                if recovery_mode == "restore_local_source":
                    report = agent.restore_local_recovery_source(
                        source_id
                    )
                    print(
                        "\nLocal graph source restoration "
                        f"for {source_id}: {report['status']}"
                    )
                    if "node_exists" in report:
                        print(f"Node already exists: {report['node_exists']}")
                        print(
                            "Recovery record already exists: "
                            f"{report['recovery_record_exists']}"
                        )
                    if "listing_count" in report:
                        print(
                            "Validated remote listing count: "
                            f"{report['listing_count']}"
                        )
                    if report["status"] == "restored":
                        print(
                            "The local node and pending recovery record "
                            "were saved atomically."
                        )
                    print(
                        "No Mem0 mutation or Gemini generation was performed.\n"
                    )
                    print_recovery_runtime_diagnostics(agent, source_id)
                    finish_timer(
                        "Command /recover-relationships",
                        command_start,
                        command_succeeded,
                    )
                    continue

                apply_recovery = recovery_mode in {"apply", "apply_source"}
                if recovery_mode == "apply_source":
                    report = agent.recover_relationships(
                        dry_run=False,
                        source_id=source_id,
                    )
                    print_recovery_runtime_diagnostics(agent, source_id)
                else:
                    report = agent.recover_relationships(
                        dry_run=not apply_recovery
                    )
                    if recovery_mode == "preview":
                        print_recovery_runtime_diagnostics(
                            agent,
                            eligibility_report=report,
                        )
                if recovery_mode == "apply_source":
                    print(
                        f"\nRelationship recovery for source {source_id}:"
                    )
                if not apply_recovery:
                    print("\nRelationship recovery preview (dry run):")
                    print(f"Eligible source IDs: {report['eligible']}")
                    print(f"Capped source IDs: {report['capped']}")
                    blocked_records = [
                        detail
                        for detail in report["details"]
                        if detail["status"] == "blocked_missing_endpoint"
                    ]
                    superseded_records = [
                        detail
                        for detail in report["details"]
                        if detail["status"] == "superseded"
                    ]

                    def print_preview_detail(detail):
                        endpoint_notes = []
                        if detail.get("missing_source_ids"):
                            endpoint_notes.append(
                                "missing source endpoint(s): "
                                f"{detail['missing_source_ids']}"
                            )
                        if detail.get("missing_target_ids"):
                            endpoint_notes.append(
                                "missing target endpoint(s): "
                                f"{detail['missing_target_ids']}"
                            )
                        if detail.get("replacement_id"):
                            endpoint_notes.append(
                                "replacement ID: "
                                f"{detail['replacement_id']}"
                            )
                        if detail.get("reason"):
                            endpoint_notes.append(detail["reason"])
                        print(
                            f"- {detail['source_id']}: {detail['status']}"
                            + (
                                f" ({'; '.join(endpoint_notes)})"
                                if endpoint_notes
                                else ""
                            )
                        )

                    print(f"\nBlocked records ({len(blocked_records)}):")
                    for detail in blocked_records:
                        print_preview_detail(detail)

                    print(
                        f"\nSuperseded historical records "
                        f"({len(superseded_records)}):"
                    )
                    for detail in superseded_records:
                        print_preview_detail(detail)
                    if report["invalid_records"]:
                        print(
                            f"Invalid preserved records: "
                            f"{report['invalid_records']}"
                        )
                    print(f"Remaining eligible work: {report['remaining']}")
                    print(f"Batch limit: {report['batch_limit']}")
                    if report.get("error"):
                        print(f"Queue error: {report['error']}")
                    print("No changes were made.\n")
                else:
                    print("\nRelationship recovery results:")
                    for key in (
                        "attempted",
                        "succeeded",
                        "succeeded_empty",
                        "pending",
                        "failed",
                        "blocked",
                    ):
                        print(f"{key.replace('_', ' ').title()}: {report[key]}")
                    for detail in report["details"]:
                        print(
                            f"- {detail['source_id']}: {detail['status']}"
                            + (
                                f" ({detail['reason']})"
                                if detail.get("reason")
                                else ""
                            )
                        )
                    if report["capped"]:
                        print(f"Capped source IDs: {report['capped']}")
                    if report["invalid_records"]:
                        print(
                            f"Invalid preserved records: "
                            f"{report['invalid_records']}"
                        )
                    if report.get("error"):
                        print(f"Recovery queue error: {report['error']}")
                    if report["remaining"]:
                        print(
                            f"Eligible records beyond this batch: "
                            f"{report['remaining']}"
                        )
                    print()
            except Exception as e:
                command_succeeded = False
                print(f"\n[ERROR] Relationship recovery failed: {e}\n")
            finish_timer(
                "Command /recover-relationships",
                command_start,
                command_succeeded,
            )
            continue

        # show knowledge graph
        if user_input.lower() == "/graph":
            command_start = start_timer()
            command_succeeded = True
            try:
                memories = agent.memory.get_all_complete(agent.user_id)

                agent.graph.sync_user_graph(
                    agent.user_id,
                    memories
                )

                user_graph = agent.graph.get_user_graph(
                    agent.user_id
                )

                nodes = user_graph.get("nodes", [])
                relationships = user_graph.get("relationships", [])

                print("\nKnowledge Graph:")
                print(f"Nodes: {len(nodes)}")
                print(f"Relationships: {len(relationships)}")

                if not nodes:
                    print("No nodes in the graph.")
                    finish_timer(
                        "Command /graph",
                        command_start,
                        command_succeeded,
                    )
                    continue

                node_number = {}

                print("\nMemories:")

                for i, node in enumerate(nodes, start=1):
                    node_number[node["id"]] = i
                    print(f"{i}. {node['memory_text']}")

                if relationships:
                    print("\nRelationships:")

                    for relationship in relationships:
                        source_number = node_number.get(
                            relationship["source"],
                            "?"
                        )

                        target_number = node_number.get(
                            relationship["target"],
                            "?"
                        )

                        print(
                            f"{source_number} --"
                            f"{relationship['relationship']}--> "
                            f"{target_number}"
                        )

                else:
                    print("\nNo relationships in the graph.")

                print()

            except Exception as e:
                command_succeeded = False
                print(
                    f"\n[ERROR] Could not retrieve knowledge graph: {e}\n"
                )

            finish_timer(
                "Command /graph",
                command_start,
                command_succeeded,
            )
            continue


        # show memories related to a specific memory
        if user_input.lower().startswith("/related"):
            command_start = start_timer()
            command_succeeded = True
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /related <memory_number>\n")
                finish_timer(
                    "Command /related",
                    command_start,
                    command_succeeded,
                )
                continue

            try:
                memory_number = int(parts[1])
            except ValueError:
                print("\nPlease enter a valid memory number.\n")
                finish_timer(
                    "Command /related",
                    command_start,
                    command_succeeded,
                )
                continue

            try:
                # Get the current Mem0 memories
                memories = agent.memory.get_all_complete(agent.user_id)

                # Synchronize the graph with Mem0
                agent.graph.sync_user_graph(
                    agent.user_id,
                    memories
                )

                user_graph = agent.graph.get_user_graph(
                    agent.user_id
                )

                nodes = user_graph.get("nodes", [])
                relationships = user_graph.get("relationships", [])

                if memory_number < 1 or memory_number > len(nodes):
                    print("\nInvalid memory number.\n")
                    finish_timer(
                        "Command /related",
                        command_start,
                        command_succeeded,
                    )
                    continue

                selected_node = nodes[memory_number - 1]
                selected_id = selected_node["id"]

                print(
                    f"\nMemories related to: "
                    f"{selected_node['memory_text']}"
                )

                print("\nRelated memories:")

                found = False

                for relationship in relationships:

                    if relationship["source"] == selected_id:
                        related_id = relationship["target"]

                    elif relationship["target"] == selected_id:
                        related_id = relationship["source"]

                    else:
                        continue

                    related_node = next(
                        (
                            node
                            for node in nodes
                            if node["id"] == related_id
                        ),
                        None
                    )

                    if related_node:
                        found = True

                        print(
                            f"- {related_node['memory_text']}"
                        )

                        print(
                            f"  Relationship: "
                            f"{relationship['relationship']}"
                        )

                if not found:
                    print("No related memories found.")

                print()

            except Exception as e:
                command_succeeded = False
                print(
                    f"\n[ERROR] Could not retrieve related memories: {e}\n"
                )

            finish_timer(
                "Command /related",
                command_start,
                command_succeeded,
            )
            continue


        # Show detailed information about a memory
        if user_input.lower().startswith("/memory-info"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /memory-info <number>\n")
                continue

            try:
                memory_number = int(parts[1])
            except ValueError:
                print("\nPlease enter a valid memory number.\n")
                continue

            try:
                active_memories = get_active_memories(agent)

                if (
                    memory_number < 1
                    or memory_number > len(active_memories)
                ):
                    print("\nInvalid memory number.\n")
                    continue

                memory = active_memories[memory_number - 1]
                metadata = memory.get("metadata", {})

                print("\nMemory details:")
                print(f"Memory: {memory['memory']}")
                print(
                    f"Importance: "
                    f"{metadata.get('importance', 'N/A')}"
                )
                print(
                    f"confidence: "
                    f"{metadata.get('confidence', 'N/A')}"
                )
                print(
                    f"Reason: "
                    f"{metadata.get('reason', 'N/A')}"
                )
                print(
                    f"Source: "
                    f"{metadata.get('source', 'N/A')}"
                )
                print(
                    f"Created: "
                    f"{memory.get('created_at', 'N/A')}"
                )
                print(
                    f"Updated: "
                    f"{memory.get('updated_at', 'N/A')}"
                )

                evolution = metadata.get("evolution")

                if isinstance(evolution, str):
                    try:
                        evolution = json.loads(evolution)
                    except (json.JSONDecodeError, TypeError):
                        evolution = None

                history = []

                if isinstance(evolution, dict):
                    history = evolution.get("history", [])

                if history:
                    print("\nTimeline:")
                    print(
                        f"  Originally: "
                        f"{history[0].get('from', 'N/A')}"
                    )

                    for change in history:
                        print(
                            f"  Updated to: "
                            f"{change.get('to', 'N/A')}"
                        )

                print()

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memory details: {e}\n"
                )

            continue

        # Show current user
        if user_input.lower() == "/user":
            print(f"\nCurrent user: {agent.user_id}\n")
            continue

        # Login as another user
        if user_input.lower().startswith("/login"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /login <user_id>\n")
                continue

            user_id = normalize_user_id(parts[1])

            if not user_id:
                print("\nUser ID cannot be empty.\n")
                continue

            try:
                new_agent = create_agent(user_id)
                save_user(user_id)
            except Exception as e:
                print(f"\n[ERROR] Could not log in as {user_id}: {e}\n")
                continue

            agent = new_agent
            print(f"\nLogged in as: {agent.user_id}\n")
            continue

        # Logout current user
        if user_input.lower() == "/logout":
            user_id = normalize_user_id(
                input("Enter your user ID: ")
            )

            if not user_id:
                print("\nLogout cancelled; user ID cannot be empty.\n")
                continue

            try:
                new_agent = create_agent(user_id)
                save_user(user_id)
            except Exception as e:
                print(f"\n[ERROR] Could not switch to {user_id}: {e}\n")
                continue

            agent = new_agent
            print(f"\nLogged in as: {agent.user_id}\n")
            continue

        # Forget memory command
        if user_input.lower().startswith("/forget"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /forget <number>\n")
                continue

            try:
                memory_number = int(parts[1])
            except ValueError:
                print("\nPlease enter a memory number.\n")
                continue

            try:
                active_memories = get_active_memories(agent)

                if (
                    memory_number < 1
                    or memory_number > len(active_memories)
                ):
                    print("\nInvalid memory number.\n")
                    continue

                memory = active_memories[memory_number - 1]
                memory_id = memory["id"]

                print(f"\nDeleting: {memory['memory']}")

                delete_result = agent.memory.delete(memory_id)
                delete_status = agent.memory.delete_result_status(
                    delete_result
                )

                if delete_status == "confirmed":
                    try:
                        agent.graph.delete_node(
                            agent.user_id,
                            memory_id
                        )
                    except Exception as e:
                        print(
                            "\n[WARNING] Mem0 deletion was confirmed, but "
                            f"graph cleanup failed: {e}\n"
                        )
                    else:
                        print("Memory deleted.\n")
                elif delete_status == "failed":
                    print(
                        "\n[ERROR] Mem0 reported that deletion failed; "
                        "the graph was left unchanged.\n"
                    )
                else:
                    print(
                        "\n[WARNING] Memory deletion could not be confirmed; "
                        "the graph was left unchanged.\n"
                    )

            except Exception as e:
                print(
                    f"\n[ERROR] Could not delete memory: {e}\n"
                )

            continue

        # Clear all memories command
        if user_input.lower() == "/clear":
            try:
                memories = agent.memory.get_all_complete(agent.user_id)

                if memories["count"] == 0:
                    print("\nNo memories to clear.\n")
                    agent.graph.sync_user_graph(
                        agent.user_id,
                        memories
                    )
                    continue

                confirm = input(
                    f"\nThis will delete {memories['count']} memories. "
                    "Are you sure? (yes/no): "
                ).strip().lower()

                if confirm == "yes":
                    result = agent.memory.clear(agent.user_id)

                    if not result["retrieval_complete"]:
                        print(
                            "\n[ERROR] Could not safely clear memories: "
                            f"{result['error']}\n"
                        )
                        continue

                    graph_synced = False
                    if result["remaining_retrieval_complete"]:
                        try:
                            agent.graph.sync_user_graph(
                                agent.user_id,
                                result["remaining_memories"]
                            )
                            graph_synced = True
                        except Exception as e:
                            print(
                                "\n[ERROR] Mem0 deletion results were recorded, "
                                "but graph synchronization failed: "
                                f"{e}\n"
                            )
                    else:
                        print(
                            "\n[WARNING] Could not verify remaining memories; "
                            "the graph was left unchanged. "
                            f"Details: {result['error']}\n"
                        )

                    if result.get("complete") and graph_synced:
                        print("\nAll memories were cleared and the graph synchronized.\n")
                    else:
                        print(
                            "\n[WARNING] Clear was partial or could not be "
                            "fully confirmed.\n"
                            f"Confirmed deleted IDs: {result['deleted_ids']}\n"
                            f"Failed IDs: {result['failed_ids']}\n"
                            f"Unconfirmed IDs: {result['unconfirmed_ids']}\n"
                            f"Remaining IDs: {result['remaining_ids']}\n"
                        )

                else:
                    print("\nMemory clearing cancelled.\n")

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memories: {e}\n"
                )

            continue

        # Help command
        if user_input.lower() == "/help":
            print(f"\n{format_command_help()}\n")
            continue

        # New conversation command
        if user_input.lower() == "/new":
            agent.conversation_history.clear()
            print("\nNew conversation started.\n")
            continue

        # Ignore empty input
        if not user_input:
            continue

        # Normal conversation
        response = agent.respond(user_input)
        print(f"\nAgent: {response}\n")
        response_metadata = getattr(agent, "last_response_metadata", {})
        if isinstance(response_metadata, dict):
            for outcome in response_metadata.get("memory_outcomes", []):
                label = MEMORY_OUTCOME_LABELS.get(outcome)
                if label:
                    print(f"[MEMORY STATUS] {label}\n")


if __name__ == "__main__":
    main()