from datetime import datetime, timezone
import math
import time
import uuid

from app.dashboard_data import _redact_sensitive_text


GENERIC_AGENT_ERROR = (
    "The agent could not complete this message. Check the service "
    "configuration and try again."
)

MEMORY_OUTCOME_LABELS = {
    "memory_saved_graph_synchronized": (
        "Memory saved and graph relationships synchronized."
    ),
    "memory_saved_graph_pending": (
        "Memory was confirmed; graph synchronization remains pending."
    ),
    "memory_succeeded_without_memory_id": (
        "Mem0 reported success without a confirmed memory ID; graph "
        "synchronization is pending reconciliation."
    ),
    "memory_saved_relationships_pending": (
        "Memory and graph node were saved; relationship recovery remains pending."
    ),
    "memory_saved_relationships_untracked": (
        "Memory and graph node were saved, but relationship recovery could "
        "not be recorded."
    ),
    "memory_replacement_cleanup_pending": (
        "The replacement memory is confirmed, but removal of the old memory "
        "is pending reconciliation."
    ),
    "memory_operation_failed": "The memory operation failed.",
    "outcome_uncertain": (
        "The write outcome is uncertain. Reconciliation is required; "
        "the write will not be repeated automatically."
    ),
}


def sanitize_conversation_text(value, limit=None):
    if not isinstance(value, str):
        return ""
    sanitized = _redact_sensitive_text(value)
    return sanitized[:limit] if limit is not None else sanitized


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _new_conversation(user_id):
    return {
        "id": uuid.uuid4().hex,
        "owner_id": user_id,
        "title": "New conversation",
        "created_at": _timestamp(),
        "messages": [],
    }


def _normalize_message(message, recover_pending=False):
    if (
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or message["role"] not in {"user", "assistant"}
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        return None, True

    normalized = {
        "role": message["role"],
        "content": sanitize_conversation_text(message["content"]),
        "timestamp": (
            message["timestamp"]
            if isinstance(message.get("timestamp"), str)
            else _timestamp()
        ),
    }
    malformed = not isinstance(message.get("timestamp"), str)

    if message["role"] == "user":
        if message.get("message_type") == "command":
            normalized["message_type"] = "command"
        status = message.get("status", "success")
        if (
            not isinstance(status, str)
            or status not in {"pending", "success", "failed"}
        ):
            status = "failed"
            malformed = True
        normalized["status"] = status
        if status == "pending" and recover_pending:
            normalized["status"] = "failed"
            normalized["error"] = (
                "This request was interrupted before a response was saved."
            )
            malformed = True
        submission_id = message.get("submission_id")
        if isinstance(submission_id, str):
            normalized["submission_id"] = submission_id
        if status == "failed":
            error = message.get("error")
            if isinstance(error, str):
                normalized["error"] = sanitize_conversation_text(error, 200)
            else:
                normalized["error"] = GENERIC_AGENT_ERROR
                malformed = True
            display_error = message.get("display_error")
            if isinstance(display_error, str) and display_error.strip():
                normalized["display_error"] = sanitize_conversation_text(
                    display_error,
                    240,
                )
    elif message.get("message_type") == "command":
        normalized["message_type"] = "command"
        command_name = message.get("command_name")
        if isinstance(command_name, str):
            normalized["command_name"] = command_name[:40]
        command_status = message.get("command_status", "success")
        if command_status in {"success", "error", "confirmation"}:
            normalized["command_status"] = command_status
        else:
            malformed = True
        command_format = message.get("command_format", "text")
        if command_format in {"text", "markdown", "code"}:
            normalized["command_format"] = command_format
        else:
            malformed = True
        if message.get("command_action") == "confirm_clear":
            normalized["command_action"] = "confirm_clear"

    latency_ms = message.get("latency_ms")
    if (
        isinstance(latency_ms, (int, float))
        and not isinstance(latency_ms, bool)
        and math.isfinite(latency_ms)
        and latency_ms >= 0
    ):
        normalized["latency_ms"] = float(latency_ms)
    elif latency_ms is not None:
        malformed = True

    activity = message.get("memory_activity")
    if isinstance(activity, list):
        normalized["memory_activity"] = [
            {
                "action": item["action"],
                "status": item["status"],
                **(
                    {"count": item["count"]}
                    if type(item.get("count")) is int
                    and item["count"] >= 0
                    else {}
                ),
            }
            for item in activity
            if (
                isinstance(item, dict)
                and isinstance(item.get("action"), str)
                and item["action"] in {
                    "retrieval",
                    "update",
                    "save",
                    "decision",
                }
                and isinstance(item.get("status"), str)
                and item["status"] in {
                    "completed",
                    "none",
                    "merged",
                    "failed",
                    "partial",
                    "pending",
                    "uncertain",
                }
            )
        ]
    elif activity is not None:
        malformed = True

    timings = message.get("timings_ms")
    if isinstance(timings, dict):
        normalized_timings = {}
        for name, duration in timings.items():
            if (
                isinstance(name, str)
                and isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and math.isfinite(duration)
                and duration >= 0
            ):
                normalized_timings[name] = float(duration)
            else:
                malformed = True
        if normalized_timings:
            normalized["timings_ms"] = normalized_timings
    elif timings is not None:
        malformed = True

    polling = message.get("polling")
    if isinstance(polling, dict):
        attempts = polling.get("attempts")
        sleep_ms = polling.get("sleep_ms")
        if type(attempts) is int and attempts >= 0:
            normalized["polling"] = {"attempts": attempts}
        else:
            malformed = True
        if (
            isinstance(sleep_ms, (int, float))
            and not isinstance(sleep_ms, bool)
            and math.isfinite(sleep_ms)
            and sleep_ms >= 0
        ):
            normalized.setdefault("polling", {})["sleep_ms"] = float(sleep_ms)
        else:
            malformed = True
    elif polling is not None:
        malformed = True

    memory_writes = message.get("memory_writes")
    if type(memory_writes) is int and memory_writes >= 0:
        normalized["memory_writes"] = memory_writes
    elif memory_writes is not None:
        malformed = True

    outcomes = message.get("memory_outcomes")
    if isinstance(outcomes, list):
        allowed_outcomes = {
            "memory_saved_graph_synchronized",
            "memory_saved_graph_pending",
            "memory_succeeded_without_memory_id",
            "memory_saved_relationships_pending",
            "memory_saved_relationships_untracked",
            "memory_replacement_cleanup_pending",
            "memory_operation_failed",
            "outcome_uncertain",
        }
        normalized["memory_outcomes"] = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, str) and outcome in allowed_outcomes
        ]
    elif outcomes is not None:
        malformed = True

    return normalized, malformed


def normalize_user_conversations(
    user_data,
    user_id,
    recover_pending=False,
):
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID must be a non-empty string.")
    user_id = user_id.strip()

    malformed = user_data is not None and not isinstance(user_data, dict)
    if not isinstance(user_data, dict):
        user_data = {"active_id": None, "conversations": []}

    raw_conversations = user_data.get("conversations", [])
    if not isinstance(raw_conversations, list):
        raw_conversations = []
        malformed = True

    conversations = []
    for raw_conversation in raw_conversations:
        if (
            not isinstance(raw_conversation, dict)
            or not isinstance(raw_conversation.get("id"), str)
            or not raw_conversation["id"].strip()
        ):
            malformed = True
            continue
        if raw_conversation.get("owner_id", user_id) != user_id:
            malformed = True
            continue

        raw_messages = raw_conversation.get("messages", [])
        if not isinstance(raw_messages, list):
            malformed = True
            raw_messages = []
        messages = []
        for raw_message in raw_messages:
            normalized, message_malformed = _normalize_message(
                raw_message,
                recover_pending,
            )
            malformed = malformed or message_malformed
            if normalized is not None:
                messages.append(normalized)

        title = raw_conversation.get("title")
        if not isinstance(title, str) or not title.strip():
            title = "New conversation"
            malformed = True
        created_at = raw_conversation.get("created_at")
        if not isinstance(created_at, str):
            created_at = _timestamp()
            malformed = True
        conversations.append(
            {
                "id": raw_conversation["id"],
                "owner_id": user_id,
                "title": sanitize_conversation_text(title, 80),
                "created_at": created_at,
                "messages": messages,
            }
        )

    active_id = user_data.get("active_id")
    if conversations and not any(
        item["id"] == active_id for item in conversations
    ):
        active_id = conversations[-1]["id"]
        malformed = malformed or user_data.get("active_id") is not None
    elif not conversations:
        active_id = None

    return {
        "active_id": active_id,
        "conversations": conversations,
    }, malformed


def start_new_conversation(user_data, user_id):
    conversation = _new_conversation(user_id)
    user_data["conversations"].append(conversation)
    user_data["active_id"] = conversation["id"]
    return conversation


def rename_conversation(conversation, title):
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Conversation title cannot be empty.")
    conversation["title"] = sanitize_conversation_text(title.strip(), 80)


def _sanitize_agent_history(agent):
    history = getattr(agent, "conversation_history", None)
    if not isinstance(history, list):
        return
    for entry in history:
        if isinstance(entry, dict) and isinstance(entry.get("text"), str):
            entry["text"] = sanitize_conversation_text(entry["text"])


def _agent_metadata(agent):
    metadata = getattr(agent, "last_response_metadata", None)
    if not isinstance(metadata, dict):
        return {"memory_activity": [], "timings_ms": {}}
    activity = metadata.get("memory_activity", [])
    timings = metadata.get("timings_ms", {})
    normalized, _ = _normalize_message(
        {
            "role": "assistant",
            "content": "metadata",
            "memory_activity": activity,
            "timings_ms": timings,
            "polling": metadata.get("polling"),
            "memory_writes": metadata.get("memory_writes"),
            "memory_outcomes": metadata.get("memory_outcomes"),
        }
    )
    return {
        "memory_activity": normalized.get("memory_activity", []),
        "timings_ms": normalized.get("timings_ms", {}),
        **(
            {"memory_writes": normalized["memory_writes"]}
            if isinstance(normalized.get("memory_writes"), int)
            else {}
        ),
        **(
            {"polling": normalized["polling"]}
            if isinstance(normalized.get("polling"), dict)
            else {}
        ),
        **(
            {"memory_outcomes": normalized.get("memory_outcomes", [])}
            if normalized.get("memory_outcomes")
            else {}
        ),
    }


def process_conversation_message(
    conversation,
    user_id,
    message,
    agent_factory,
    submission_id=None,
    persist=None,
):
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID must be a non-empty string.")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty.")

    safe_message = sanitize_conversation_text(message.strip())
    if not safe_message:
        raise ValueError("Message cannot be empty.")
    if conversation.get("owner_id") != user_id.strip():
        raise ValueError("Conversation does not belong to the selected user.")

    if submission_id is not None and any(
        entry.get("submission_id") == submission_id
        for entry in conversation["messages"]
        if isinstance(entry, dict)
    ):
        return {"success": False, "duplicate": True}

    user_entry = {
        "role": "user",
        "content": safe_message,
        "timestamp": _timestamp(),
        "status": "pending",
    }
    if isinstance(submission_id, str):
        user_entry["submission_id"] = submission_id
    conversation["messages"].append(user_entry)
    if conversation["title"] == "New conversation":
        conversation["title"] = safe_message[:80]

    agent = None
    try:
        if persist is not None:
            try:
                persist(conversation)
            except Exception as error:
                user_entry["status"] = "failed"
                user_entry["display_error"] = (
                    "This message was not sent because local history "
                    "could not be saved."
                )
                user_entry["error"] = sanitize_conversation_text(
                    str(error),
                    200,
                )
                return {
                    "success": False,
                    "error": user_entry["display_error"],
                    "technical_error": user_entry["error"],
                    "persistence_failed": True,
                }
        agent = agent_factory()
        if getattr(agent, "user_id", None) != user_id.strip():
            raise ValueError("Agent user does not match the selected user.")

        started = time.perf_counter()
        response = agent.respond(safe_message)
        latency_ms = (time.perf_counter() - started) * 1000
        if not isinstance(response, str) or not response.strip():
            raise ValueError("The agent returned an empty response.")

        safe_response = sanitize_conversation_text(response.strip())
        if not safe_response:
            raise ValueError("The agent returned an empty response.")

        user_entry["status"] = "success"
        metadata = _agent_metadata(agent)
        assistant_entry = {
            "role": "assistant",
            "content": safe_response,
            "timestamp": _timestamp(),
            "latency_ms": latency_ms,
            **metadata,
        }
        conversation["messages"].append(assistant_entry)
        if persist is not None:
            try:
                persist(conversation)
            except Exception as error:
                conversation["messages"].pop()
                user_entry["status"] = "failed"
                user_entry["error"] = (
                    "The reply was generated but could not be saved locally."
                )
                user_entry["display_error"] = (
                    "The answer was generated but could not be saved to "
                    "local history."
                )
                return {
                    "success": False,
                    "error": user_entry["error"],
                    "technical_error": sanitize_conversation_text(
                        str(error), 200
                    ),
                    "response": safe_response,
                    "persistence_failed": True,
                }
        return {
            "success": True,
            "response": safe_response,
            "latency_ms": latency_ms,
        }
    except Exception as error:
        user_entry["status"] = "failed"
        user_entry["error"] = sanitize_conversation_text(str(error), 200)
        user_entry["display_error"] = GENERIC_AGENT_ERROR
        if agent is not None:
            user_entry.update(_agent_metadata(agent))
        persistence_failed = False
        if persist is not None:
            try:
                persist(conversation)
            except Exception as persistence_error:
                user_entry["persistence_failed"] = True
                user_entry["persistence_error"] = (
                    sanitize_conversation_text(str(persistence_error), 200)
                )
                persistence_failed = True
        return {
            "success": False,
            "error": GENERIC_AGENT_ERROR,
            "technical_error": user_entry["error"],
            "persistence_failed": persistence_failed,
        }
    finally:
        if agent is not None:
            _sanitize_agent_history(agent)


def process_conversation_command(
    conversation,
    user_id,
    message,
    result,
    submission_id=None,
    persist=None,
):
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("User ID must be a non-empty string.")
    if (
        conversation.get("owner_id") != user_id.strip()
        or not isinstance(result, dict)
        or result.get("handled") is not True
    ):
        raise ValueError("Command result does not belong to this conversation.")
    if submission_id is not None and any(
        entry.get("submission_id") == submission_id
        for entry in conversation["messages"]
        if isinstance(entry, dict)
    ):
        return {"success": False, "duplicate": True}

    timestamp = _timestamp()
    user_entry = {
        "role": "user",
        "content": sanitize_conversation_text(message.strip()),
        "timestamp": timestamp,
        "status": "success",
        "message_type": "command",
    }
    if isinstance(submission_id, str):
        user_entry["submission_id"] = submission_id
    assistant_entry = {
        "role": "assistant",
        "content": sanitize_conversation_text(result.get("text", "")),
        "timestamp": _timestamp(),
        "message_type": "command",
        "command_name": result.get("command", "command"),
        "command_status": result.get("status", "success"),
        "command_format": result.get("format", "text"),
    }
    action = result.get("action")
    if isinstance(action, dict) and action.get("type") == "confirm_clear":
        assistant_entry["command_action"] = "confirm_clear"

    conversation["messages"].extend((user_entry, assistant_entry))
    if conversation["title"] == "New conversation":
        conversation["title"] = user_entry["content"][:80]
    try:
        if persist is not None:
            persist(conversation)
    except Exception as error:
        user_entry["status"] = "failed"
        user_entry["error"] = sanitize_conversation_text(str(error), 200)
        user_entry["display_error"] = (
            "The command ran, but its response could not be saved locally."
        )
        return {
            "success": False,
            "persistence_failed": True,
            "error": user_entry["display_error"],
            "technical_error": user_entry["error"],
        }
    return {"success": True, "response": assistant_entry["content"]}
