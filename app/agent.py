import math
import os
import logging
import json
import copy
import hashlib
import time
from platform import node
import traceback


from app.performance import timed_operation
from app.recovery import evaluate_relationship_recovery

logging.getLogger("google").setLevel(logging.ERROR)

from google import genai
from google.genai import types

from app.memory import MemoryManager
from app.graph import KnowledgeGraph


def _event_poll_details(metrics):
    return (
        f"attempts={metrics.get('attempts', 0)}, "
        f"request_ms={metrics.get('request_ms', 0.0):.2f}, "
        f"sleep_ms={metrics.get('sleep_ms', 0.0):.2f}, "
        f"outcome={metrics.get('outcome', 'unknown')}"
    )


class Agent:
    """AI agent with long-term memory."""

    RELATIONSHIP_RECOVERY_BATCH_LIMIT = 5
    RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT = 3
    RELATIONSHIP_LABELS = {
        "has_backend",
        "uses_backend",
        "has_milestone",
        "has_deadline",
        "prerequisite_for",
        "has_prerequisite",
        "developed_on",
        "runs_on",
        "uses",
        "implements",
        "reports_performance_of",
        "demonstrates_generalization_of",
        "applies_to",
        "part_of",
        "member_of",
        "located_in",
        "studies",
        "prefers",
        "interested_in",
        "motivated_by",
        "describes",
        "measured_on",
    }
    CONTEXT_REFERENCE_WORDS = {
        "it",
        "that",
        "this",
        "same",
        "other",
        "one",
        "them",
        "there",
        "he",
        "she",
        "they",
    }

    @classmethod
    def _needs_memory_fallback(cls, message):
        words = {
            word.strip(".,!?;:'\"()[]{}").casefold()
            for word in message.split()
        }
        return (
            len(words) <= 8
            and bool(words & cls.CONTEXT_REFERENCE_WORDS)
        )

    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = MemoryManager()
        self.graph = KnowledgeGraph()
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please add your api key in the environment."
            )

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30000, 
                retry_options= types.HttpRetryOptions(
                    attempts=1
                )
            )
        )

        self.conversation_history = []
        self.max_history = 10
        self.last_response_metadata = {
            "memory_activity": [],
            "timings_ms": {},
        }

    def _record_memory_activity(self, action, status, count=None):
        metadata = getattr(self, "last_response_metadata", None)
        if not isinstance(metadata, dict):
            return
        activity = metadata.setdefault("memory_activity", [])
        event = {"action": action, "status": status}
        if isinstance(count, int) and count >= 0:
            event["count"] = count
        activity.append(event)

    def _record_response_timing(self, name, started_at):
        metadata = getattr(self, "last_response_metadata", None)
        if not isinstance(metadata, dict):
            return
        timings = metadata.setdefault("timings_ms", {})
        timings[name] = timings.get(name, 0.0) + (
            time.perf_counter() - started_at
        ) * 1000

    @staticmethod
    def _normalize_confidence(value, default=0.5):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return default

        return max(0.0, min(confidence, 1.0))

    @staticmethod
    def _delete_succeeded(result):
        return MemoryManager.delete_result_status(result) == "confirmed"

    @staticmethod
    def _parse_evolution(evolution):
        if isinstance(evolution, str):
            try:
                return json.loads(evolution)
            except (json.JSONDecodeError, TypeError):
                return None

        return evolution

    @classmethod
    def _historical_memory_ids(cls, evolution):
        evolution = cls._parse_evolution(evolution)
        historical_ids = set()

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

            evolution = evolution.get("legacy_entries", [])

        if isinstance(evolution, list):
            for entry in evolution:
                if isinstance(entry, str) and entry.startswith(
                    "previous_memory_id."
                ):
                    historical_ids.add(entry.split(".", 1)[1])
                elif isinstance(entry, dict) and entry.get("memory_id"):
                    historical_ids.add(entry["memory_id"])

        return historical_ids

    @classmethod
    def _merge_deduplication_metadata(
        cls,
        source_memories,
        importance,
        confidence,
        reason
    ):
        merged_metadata = {}
        history = []
        legacy_entries = []
        legacy_values = []
        source_evolutions = []
        has_evolution = False

        for memory in source_memories:
            metadata = memory.get("metadata", {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key != "evolution" and key not in merged_metadata:
                        merged_metadata[key] = value

                if "evolution" not in metadata:
                    continue

                has_evolution = True
                raw_evolution = metadata["evolution"]
                source_evolution = {
                    "memory_id": memory.get("id"),
                    "value": raw_evolution
                }
                if source_evolution not in source_evolutions:
                    source_evolutions.append(source_evolution)

                evolution = cls._parse_evolution(raw_evolution)
                if isinstance(evolution, dict):
                    existing_history = evolution.get("history", [])
                    if isinstance(existing_history, list):
                        for entry in existing_history:
                            if entry not in history:
                                history.append(entry)

                    if evolution.get("previous_memory"):
                        previous_entry = {
                            "memory_id": evolution.get(
                                "previous_memory_id"
                            ),
                            "from": evolution["previous_memory"],
                            "to": memory.get("memory", "")
                        }
                        if previous_entry not in history:
                            history.insert(0, previous_entry)

                    existing_legacy_entries = evolution.get(
                        "legacy_entries",
                        []
                    )
                    if isinstance(existing_legacy_entries, list):
                        for entry in existing_legacy_entries:
                            if entry not in legacy_entries:
                                legacy_entries.append(entry)
                elif isinstance(evolution, list):
                    for entry in evolution:
                        if entry not in legacy_entries:
                            legacy_entries.append(entry)
                elif isinstance(raw_evolution, str):
                    if raw_evolution not in legacy_values:
                        legacy_values.append(raw_evolution)
        merged_metadata.update(
            {
                "importance": importance,
                "confidence": confidence,
                "reason": reason,
            }
        )
        merged_metadata["source_memories"] = [
            dict(memory) for memory in source_memories
        ]

        if has_evolution:
            evolution_metadata = {}
            if history:
                evolution_metadata["history"] = history
            if legacy_entries:
                evolution_metadata["legacy_entries"] = legacy_entries
            if legacy_values:
                evolution_metadata["legacy_values"] = legacy_values
                if len(legacy_values) == 1:
                    evolution_metadata["legacy_value"] = legacy_values[0]
            if source_evolutions:
                evolution_metadata["source_evolutions"] = source_evolutions
            merged_metadata["evolution"] = json.dumps(evolution_metadata)

        return merged_metadata

    def get_memory_id(
        self,
        event_id,
        attempts=None,
        delay=None,
        timeout_seconds=None,
    ):
        result = self.resolve_memory_event(
            event_id,
            attempts=attempts,
            delay=delay,
            timeout_seconds=timeout_seconds,
        )
        self._last_event_resolution = result
        return result["memory_id"]

    def resolve_memory_event(
        self,
        event_id,
        attempts=None,
        delay=None,
        timeout_seconds=None,
    ):
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("A valid Mem0 event ID is required.")
        try:
            configured_timeout = float(
                os.getenv("MEM0_EVENT_TIMEOUT_SECONDS", "8")
            )
            configured_interval = float(
                os.getenv("MEM0_EVENT_POLL_INTERVAL_SECONDS", "0.25")
            )
            configured_max_interval = float(
                os.getenv("MEM0_EVENT_POLL_MAX_INTERVAL_SECONDS", "1")
            )
        except ValueError as error:
            raise ValueError(
                "Mem0 event polling environment settings must be numbers."
            ) from error
        timeout_seconds = (
            configured_timeout if timeout_seconds is None else timeout_seconds
        )
        interval = configured_interval if delay is None else delay
        max_attempts = attempts if attempts is not None else 120
        try:
            finite_intervals = all(
                math.isfinite(float(value))
                for value in (
                    timeout_seconds,
                    interval,
                    configured_max_interval,
                )
            )
        except (OverflowError, TypeError, ValueError):
            finite_intervals = False
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds < 0
            or not finite_intervals
            or not isinstance(interval, (int, float))
            or isinstance(interval, bool)
            or interval <= 0
            or not isinstance(configured_max_interval, (int, float))
            or configured_max_interval <= 0
            or configured_max_interval < interval
            or type(max_attempts) is not int
            or max_attempts < 1
        ):
            raise ValueError("Mem0 event polling configuration is invalid.")

        deadline = time.monotonic() + timeout_seconds
        next_interval = float(interval)
        metrics = {
            "attempts": 0,
            "request_ms": 0.0,
            "sleep_ms": 0.0,
            "outcome": "timeout",
            "memory_id": None,
        }
        pending_statuses = {
            "PENDING",
            "RUNNING",
            "PROCESSING",
            "IN_PROGRESS",
        }
        started_at = time.perf_counter()
        with timed_operation("Mem0 event polling total") as timing:
            for attempt in range(1, max_attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                metrics["attempts"] = attempt
                request_start = time.perf_counter()
                request_ms = 0.0
                request_status = "exception"
                result_category = "exception"
                try:
                    response = self.memory.get_event(
                        event_id,
                        timeout_seconds=remaining,
                    )
                    request_ms = (time.perf_counter() - request_start) * 1000
                    metrics["request_ms"] += request_ms
                    request_status = (
                        "success"
                        if response.status_code == 200
                        else f"http_{response.status_code}"
                    )
                    if response.status_code != 200:
                        result_category = "http_error"
                    else:
                        event = response.json()
                        status = (
                            event.get("status")
                            if isinstance(event, dict)
                            else None
                        )
                        normalized_status = (
                            status.upper() if isinstance(status, str) else ""
                        )
                        if normalized_status == "SUCCEEDED":
                            results = event.get("results", [])
                            memory_id = next(
                                (
                                    item.get("id")
                                    for item in results
                                    if (
                                        isinstance(item, dict)
                                        and isinstance(item.get("id"), str)
                                        and item["id"].strip()
                                    )
                                ),
                                None,
                            ) if isinstance(results, list) else None
                            if memory_id:
                                result_category = "succeeded_with_memory_id"
                                metrics["outcome"] = "succeeded"
                                metrics["memory_id"] = memory_id
                            else:
                                result_category = "succeeded_without_memory_id"
                                metrics["outcome"] = (
                                    "succeeded_without_memory_id"
                                )
                            self._record_response_timing(
                                "memory_event_polling",
                                started_at,
                            )
                            self._record_poll_metrics(metrics)
                            timing["succeeded"] = bool(memory_id)
                            timing["details"] = _event_poll_details(metrics)
                            self._log_event_poll(
                                attempt,
                                request_status,
                                result_category,
                                request_ms,
                                0,
                            )
                            return {
                                "status": metrics["outcome"],
                                "event_id": event_id,
                                "memory_id": memory_id,
                                **metrics,
                            }
                        if normalized_status == "FAILED":
                            result_category = "failed"
                            metrics["outcome"] = "failed"
                            self._record_response_timing(
                                "memory_event_polling",
                                started_at,
                            )
                            self._record_poll_metrics(metrics)
                            timing["succeeded"] = False
                            timing["details"] = _event_poll_details(metrics)
                            self._log_event_poll(
                                attempt,
                                request_status,
                                result_category,
                                request_ms,
                                0,
                            )
                            return {
                                "status": "failed",
                                "event_id": event_id,
                                **metrics,
                            }
                        result_category = (
                            "pending"
                            if normalized_status in pending_statuses
                            else "unknown_status"
                        )
                except Exception as error:
                    if request_ms == 0.0:
                        request_ms = (
                            time.perf_counter() - request_start
                        ) * 1000
                        metrics["request_ms"] += request_ms
                    result_category = (
                        f"exception_{type(error).__name__}"
                    )

                remaining = deadline - time.monotonic()
                sleep_ms = 0.0
                if remaining > 0 and attempt < max_attempts:
                    sleep_seconds = min(next_interval, remaining)
                    sleep_start = time.perf_counter()
                    time.sleep(sleep_seconds)
                    sleep_ms = (time.perf_counter() - sleep_start) * 1000
                    metrics["sleep_ms"] += sleep_ms
                    next_interval = min(
                        next_interval * 2,
                        configured_max_interval,
                    )
                self._log_event_poll(
                    attempt,
                    request_status,
                    result_category,
                    request_ms,
                    sleep_ms,
                )
                if remaining <= 0 or attempt >= max_attempts:
                    break

            metrics["outcome"] = "timeout"
            timing["succeeded"] = False
            timing["details"] = _event_poll_details(metrics)

        self._record_response_timing("memory_event_polling", started_at)
        self._record_poll_metrics(metrics)
        print("[GRAPH] Timed out waiting for Mem0 event; reconciliation is required.")
        return {
            "status": "timeout",
            "event_id": event_id,
            **metrics,
        }

    def _record_poll_metrics(self, metrics):
        metadata = getattr(self, "last_response_metadata", None)
        if isinstance(metadata, dict):
            polling = metadata.setdefault("polling", {})
            polling["attempts"] = polling.get("attempts", 0) + metrics["attempts"]
            polling["sleep_ms"] = polling.get("sleep_ms", 0.0) + metrics["sleep_ms"]

    def _record_memory_outcome(self, status):
        metadata = getattr(self, "last_response_metadata", None)
        if isinstance(metadata, dict):
            metadata.setdefault("memory_outcomes", []).append(status)

    @staticmethod
    def _log_event_poll(attempt, http_status, result, request_ms, sleep_ms):
        print(
            f"[TIME] Mem0 event poll attempt {attempt}: "
            f"http={http_status}, result={result}, "
            f"request_ms={request_ms:.2f}, sleep_ms={sleep_ms:.2f}"
        )

    def _pending_mem0_operations(self):
        get_pending = getattr(self.graph, "get_pending_mem0_operations", None)
        if not callable(get_pending):
            return []
        return get_pending(self.user_id)

    def _record_mem0_operation(
        self,
        event_id,
        memory_text,
        status,
        operation_type,
        related_memory_ids,
        memory_id=None,
    ):
        record = getattr(self.graph, "record_mem0_operation", None)
        if callable(record):
            return record(
                self.user_id,
                event_id,
                memory_text,
                status,
                operation_type=operation_type,
                related_memory_ids=related_memory_ids,
                memory_id=memory_id,
            )
        return None

    def _update_mem0_operation(
        self,
        operation_id,
        status,
        memory_id=None,
        deleted_related_memory_ids=None,
    ):
        update = getattr(self.graph, "update_mem0_operation", None)
        if callable(update) and operation_id:
            return update(
                self.user_id,
                operation_id,
                status,
                memory_id=memory_id,
                deleted_related_memory_ids=deleted_related_memory_ids,
            )
        return False

    def _commit_mem0_operation(
        self,
        operation,
        memory_id,
        retain_operation=False,
    ):
        reconcile = getattr(self.graph, "reconcile_mem0_operation", None)
        if callable(reconcile):
            return reconcile(
                self.user_id,
                operation["operation_id"],
                memory_id,
                retain_operation=retain_operation,
            )
        self.graph.add_node(
            self.user_id,
            memory_id,
            operation["memory_text"],
        )
        return True

    def _finish_confirmed_mem0_operation(self, operation, memory_id):
        operation_id = operation["operation_id"]
        replacement_operation = operation["operation_type"] in {
            "update",
            "merge",
        }
        graph_started = time.perf_counter()
        try:
            if not self._update_mem0_operation(
                operation_id,
                "succeeded",
                memory_id=memory_id,
            ):
                return "graph_synchronization_pending"
            synced = self._commit_mem0_operation(
                operation,
                memory_id,
                retain_operation=replacement_operation,
            )
        except Exception as error:
            print(
                "[MEMORY] Mem0 memory was confirmed, but graph "
                f"synchronization is pending: {error}"
            )
            return "graph_synchronization_pending"
        finally:
            self._record_response_timing(
                "graph_persistence",
                graph_started,
            )
        if not synced:
            return "graph_synchronization_pending"

        if replacement_operation:
            deleted_ids = set(
                operation.get("deleted_related_memory_ids", [])
            )
            for related_id in operation["related_memory_ids"]:
                if related_id in deleted_ids:
                    continue
                if related_id == memory_id:
                    deleted_ids.add(related_id)
                    graph_started = time.perf_counter()
                    try:
                        progress_saved = self._update_mem0_operation(
                            operation_id,
                            "succeeded",
                            memory_id=memory_id,
                            deleted_related_memory_ids=sorted(deleted_ids),
                        )
                    finally:
                        self._record_response_timing(
                            "graph_persistence",
                            graph_started,
                        )
                    if not progress_saved:
                        return "replacement_cleanup_pending"
                    continue
                try:
                    deleted = self.memory.delete(related_id)
                except Exception as error:
                    print(
                        "[MEMORY] Replacement memory is confirmed, but "
                        f"deletion of related memory {related_id} failed: "
                        f"{error}"
                    )
                    return "replacement_cleanup_pending"
                if not self._delete_succeeded(deleted):
                    print(
                        "[MEMORY] Replacement memory is confirmed, but "
                        f"deletion of related memory {related_id} was not "
                        "confirmed."
                    )
                    return "replacement_cleanup_pending"
                deleted_ids.add(related_id)
                graph_started = time.perf_counter()
                try:
                    progress_saved = self._update_mem0_operation(
                        operation_id,
                        "succeeded",
                        memory_id=memory_id,
                        deleted_related_memory_ids=sorted(deleted_ids),
                    )
                finally:
                    self._record_response_timing(
                        "graph_persistence",
                        graph_started,
                    )
                if not progress_saved:
                    print(
                        "[MEMORY] Related memory deletion was confirmed, "
                        "but its reconciliation state could not be saved."
                    )
                    return "replacement_cleanup_pending"

            reconcile_replacement = getattr(
                self.graph,
                "reconcile_mem0_replacement_operation",
                None,
            )
            if not callable(reconcile_replacement):
                return "graph_synchronization_pending"
            try:
                graph_started = time.perf_counter()
                replacement_result = reconcile_replacement(
                    self.user_id,
                    operation_id,
                    memory_id,
                )
            except Exception as error:
                self._record_response_timing(
                    "graph_persistence",
                    graph_started,
                )
                print(
                    "[MEMORY] Related Mem0 memories were deleted, but "
                    f"atomic graph replacement is pending: {error}"
                )
                return "graph_synchronization_pending"
            self._record_response_timing(
                "graph_persistence",
                graph_started,
            )
            if replacement_result.get("status") != "reconciled":
                return "graph_synchronization_pending"

        try:
            relationships_synced = self.create_memory_relationships(
                memory_id,
                operation["memory_text"],
            )
        except Exception as error:
            print(
                "[MEMORY] Memory and graph node are saved, but relationship "
                f"synchronization failed: {error}"
            )
            return "memory_saved_relationships_pending"
        if relationships_synced:
            return "synchronized"
        return (
            "memory_saved_relationships_untracked"
            if getattr(self, "_last_relationship_status", "pending")
            == "untracked"
            else "memory_saved_relationships_pending"
        )

    def _write_memory_and_sync(
        self,
        messages,
        memory_text,
        metadata,
        operation_type="add",
        related_memory_ids=None,
        timeout_seconds=None,
        create_relationships=True,
    ):
        response_metadata = getattr(self, "last_response_metadata", None)
        if isinstance(response_metadata, dict):
            response_metadata["memory_writes"] = (
                response_metadata.get("memory_writes", 0) + 1
            )
        add_started = time.perf_counter()
        try:
            add_result = self.memory.add(
                messages,
                self.user_id,
                metadata=metadata,
            )
        except Exception as error:
            self._record_response_timing("mem0_add", add_started)
            try:
                self._record_mem0_operation(
                    None,
                    memory_text,
                    "outcome_uncertain",
                    operation_type,
                    related_memory_ids,
                )
            except Exception as ledger_error:
                print(
                    "[MEMORY] Mem0 add raised an error and its uncertain "
                    f"outcome could not be persisted: {ledger_error}"
                )
            self._record_memory_activity("save", "uncertain", 1)
            self._record_memory_outcome("outcome_uncertain")
            print(
                "[MEMORY] Mem0 add outcome is uncertain; no retry was made. "
                f"Diagnostic: {error}"
            )
            return {"status": "outcome_uncertain"}
        self._record_response_timing("mem0_add", add_started)

        if not isinstance(add_result, dict) or not add_result:
            try:
                self._record_mem0_operation(
                    None,
                    memory_text,
                    "outcome_uncertain",
                    operation_type,
                    related_memory_ids,
                )
            except Exception as error:
                print(
                    "[MEMORY] Add result was incomplete and its uncertain "
                    f"outcome could not be persisted: {error}"
                )
            self._record_memory_activity("save", "uncertain", 1)
            self._record_memory_outcome("outcome_uncertain")
            print(
                "[MEMORY] Mem0 returned no verifiable event ID; outcome is "
                "uncertain and the write will not be retried automatically."
            )
            return {"status": "outcome_uncertain"}

        event_id = add_result.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            try:
                self._record_mem0_operation(
                    None,
                    memory_text,
                    "outcome_uncertain",
                    operation_type,
                    related_memory_ids,
                )
            except Exception as error:
                print(
                    "[MEMORY] Mem0 event ID was missing and its uncertain "
                    f"outcome could not be persisted: {error}"
                )
            self._record_memory_activity("save", "uncertain", 1)
            self._record_memory_outcome("outcome_uncertain")
            print(
                "[MEMORY] Mem0 accepted a write without an event ID; outcome "
                "is uncertain and it will not be retried automatically."
            )
            return {"status": "outcome_uncertain"}

        try:
            operation = self._record_mem0_operation(
                event_id,
                memory_text,
                "pending",
                operation_type,
                related_memory_ids,
            )
        except Exception as error:
            operation = None
            print(
                "[MEMORY] Could not persist Mem0 event for reconciliation; "
                f"event status will still be checked: {error}"
            )

        self._last_event_resolution = None
        try:
            if timeout_seconds is None:
                memory_id = self.get_memory_id(event_id)
            else:
                memory_id = self.get_memory_id(
                    event_id,
                    timeout_seconds=timeout_seconds,
                )
            resolution = getattr(self, "_last_event_resolution", None)
            status = (
                resolution["status"]
                if isinstance(resolution, dict)
                else ("succeeded" if memory_id else "timeout")
            )
        except Exception as error:
            memory_id = None
            status = "timeout"
            print(
                "[MEMORY] Mem0 event polling failed; event will be "
                f"reconciled later: {error}"
            )

        if not memory_id:
            operation_status = (
                "succeeded_without_memory_id"
                if status == "succeeded_without_memory_id"
                else "failed"
                if status == "failed"
                else "pending"
            )
            if operation:
                graph_started = time.perf_counter()
                try:
                    self._update_mem0_operation(
                        operation["operation_id"],
                        operation_status,
                    )
                except Exception as error:
                    print(
                        "[MEMORY] Could not update pending Mem0 operation "
                        f"state: {error}"
                    )
            activity_status = (
                "failed"
                if status == "failed"
                else "pending"
                if status == "succeeded_without_memory_id"
                else "uncertain"
            )
            self._record_memory_activity("save", activity_status, 1)
            self._record_memory_outcome(
                "memory_operation_failed"
                if status == "failed"
                else "memory_succeeded_without_memory_id"
                if status == "succeeded_without_memory_id"
                else "outcome_uncertain"
            )
            print(
                "[MEMORY] "
                + (
                    "Mem0 event failed; no graph node was created."
                    if status == "failed"
                    else "Mem0 event succeeded without a confirmed memory ID; "
                    "graph synchronization is pending."
                    if status == "succeeded_without_memory_id"
                    else "Mem0 operation outcome is uncertain; event "
                    "reconciliation is required and the write will not be "
                    "repeated."
                )
            )
            return {
                "status": status,
                "event_id": event_id,
                "operation": operation,
            }

        graph_started = time.perf_counter()
        if operation:
            try:
                self._update_mem0_operation(
                    operation["operation_id"],
                    "succeeded",
                    memory_id=memory_id,
                )
                synced = self._commit_mem0_operation(operation, memory_id)
            except Exception as error:
                synced = False
                print(
                    "[MEMORY] Mem0 memory was confirmed, but graph "
                    f"synchronization is pending: {error}"
                )
            finally:
                self._record_response_timing(
                    "graph_persistence",
                    graph_started,
                )
        else:
            try:
                self.graph.add_node(self.user_id, memory_id, memory_text)
                synced = True
            except Exception as error:
                synced = False
                print(
                    "[MEMORY] Mem0 memory was confirmed, but graph node "
                    f"creation failed; synchronization is pending: {error}"
                )
            finally:
                self._record_response_timing(
                    "graph_persistence",
                    graph_started,
                )

        if not synced:
            self._record_memory_activity("save", "partial", 1)
            self._record_memory_outcome("memory_saved_graph_pending")
            return {
                "status": "saved_graph_pending",
                "event_id": event_id,
                "memory_id": memory_id,
                "operation": operation,
            }
        if create_relationships:
            try:
                relationships_synced = self.create_memory_relationships(
                    memory_id,
                    memory_text,
                )
            except Exception as error:
                relationships_synced = False
                print(
                    "[MEMORY] Mem0 memory and graph node were saved, but "
                    f"relationship synchronization failed: {error}"
                )
            if relationships_synced:
                self._record_memory_activity("save", "completed", 1)
                self._record_memory_outcome(
                    "memory_saved_graph_synchronized"
                )
            else:
                relationship_status = getattr(
                    self,
                    "_last_relationship_status",
                    "pending",
                )
                self._record_memory_activity("save", "partial", 1)
                self._record_memory_outcome(
                    (
                        "memory_saved_relationships_untracked"
                        if relationship_status == "untracked"
                        else "memory_saved_relationships_pending"
                    )
                )
                print(
                    "[MEMORY] Graph relationship persistence failed; the "
                    "Mem0 memory and graph node were saved. "
                    + (
                        "Recovery tracking could not be recorded."
                        if relationship_status == "untracked"
                        else "Relationship recovery remains pending."
                    )
                )
            if not relationships_synced:
                return {
                    "status": (
                        "saved_relationships_untracked"
                        if getattr(
                            self,
                            "_last_relationship_status",
                            "pending",
                        ) == "untracked"
                        else "saved_relationships_pending"
                    ),
                    "event_id": event_id,
                    "memory_id": memory_id,
                    "operation": operation,
                }
        return {
            "status": "saved_graph_synced",
            "event_id": event_id,
            "memory_id": memory_id,
            "operation": operation,
        }

    def reconcile_pending_memory_events(
        self,
        limit=10,
        timeout_seconds=1.5,
    ):
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer.")
        operations = self._pending_mem0_operations()[:limit]
        report = {
            "checked": 0,
            "synchronized": 0,
            "pending": 0,
            "untracked": 0,
            "failed": 0,
            "details": [],
        }
        for operation in operations:
            report["checked"] += 1
            operation_id = operation["operation_id"]
            event_id = operation.get("event_id")
            memory_id = (
                operation.get("memory_id")
                if operation.get("status") == "succeeded"
                else None
            )
            status = operation["status"]
            if not memory_id and event_id:
                resolution = self.resolve_memory_event(
                    event_id,
                    timeout_seconds=timeout_seconds,
                )
                status = resolution["status"]
                memory_id = resolution.get("memory_id")
            if status == "failed":
                self._update_mem0_operation(operation_id, "failed")
                report["failed"] += 1
                report["details"].append(
                    {"status": "failed", "operation_type": operation["operation_type"]}
                )
                continue
            if not memory_id:
                if event_id and status in {
                    "succeeded_without_memory_id",
                    "timeout",
                }:
                    self._update_mem0_operation(
                        operation_id,
                        (
                            "succeeded_without_memory_id"
                            if status == "succeeded_without_memory_id"
                            else "pending"
                        ),
                    )
                report["pending"] += 1
                report["details"].append(
                    {
                        "status": (
                            "succeeded_without_memory_id"
                            if status == "succeeded_without_memory_id"
                            else "outcome_uncertain"
                        ),
                        "operation_type": operation["operation_type"],
                    }
                )
                continue
            result_status = self._finish_confirmed_mem0_operation(
                operation,
                memory_id,
            )
            if result_status == "synchronized":
                report["synchronized"] += 1
                report["details"].append(
                    {
                        "status": "memory_saved_graph_synchronized",
                        "operation_type": operation["operation_type"],
                    }
                )
            elif result_status == "memory_saved_relationships_untracked":
                report["untracked"] += 1
                report["details"].append(
                    {
                        "status": result_status,
                        "operation_type": operation["operation_type"],
                    }
                )
            else:
                report["pending"] += 1
                report["details"].append(
                    {
                        "status": result_status,
                        "operation_type": operation["operation_type"],
                    }
                )
        return report

    @staticmethod
    def _normalized_memory_text(value):
        return " ".join(value.casefold().split())

    def _reconcile_matching_memory_operation(
        self,
        memory_text,
        timeout_seconds=1.5,
        operation_types=None,
    ):
        normalized_text = self._normalized_memory_text(memory_text)
        for operation in self._pending_mem0_operations():
            if (
                operation_types is not None
                and operation["operation_type"] not in operation_types
            ):
                continue
            if (
                self._normalized_memory_text(operation["memory_text"])
                != normalized_text
            ):
                continue
            operation_id = operation["operation_id"]
            event_id = operation.get("event_id")
            memory_id = (
                operation.get("memory_id")
                if operation.get("status") == "succeeded"
                else None
            )
            status = operation["status"]
            if not memory_id and event_id:
                resolution = self.resolve_memory_event(
                    event_id,
                    timeout_seconds=timeout_seconds,
                )
                status = resolution["status"]
                memory_id = resolution.get("memory_id")
            if status == "failed":
                self._update_mem0_operation(operation_id, "failed")
                continue
            if not memory_id:
                if event_id:
                    self._update_mem0_operation(
                        operation_id,
                        (
                            "succeeded_without_memory_id"
                            if status == "succeeded_without_memory_id"
                            else "pending"
                        ),
                    )
                return "pending"
            result_status = self._finish_confirmed_mem0_operation(
                operation,
                memory_id,
            )
            if result_status == "synchronized":
                self._record_memory_activity("save", "completed", 1)
                self._record_memory_outcome("memory_saved_graph_synchronized")
                return "synchronized"
            if result_status == "memory_saved_relationships_untracked":
                self._record_memory_activity("save", "partial", 1)
                self._record_memory_outcome(
                    "memory_saved_relationships_untracked"
                )
                return "untracked"
            if result_status in {
                "graph_synchronization_pending",
                "replacement_cleanup_pending",
                "memory_saved_relationships_pending",
            }:
                self._record_memory_activity("save", "partial", 1)
                outcome = {
                    "graph_synchronization_pending":
                        "memory_saved_graph_pending",
                    "replacement_cleanup_pending":
                        "memory_replacement_cleanup_pending",
                    "memory_saved_relationships_pending":
                        "memory_saved_relationships_pending",
                }[result_status]
                self._record_memory_outcome(outcome)
                return "partial"
            self._record_memory_activity("save", "uncertain", 1)
            self._record_memory_outcome("outcome_uncertain")
            return "pending"
        return None

    def remember(self, message, response):
        """
        Extract important long-term information from the user's message.
        Before storing each extracted memory, check for duplicate or
        highly similar existing memories and merge them when necessary.
        """

        prompt = f"""
You are a long-term memory extraction system for an AI agent.

Your job is to extract information from the user's message that
could be useful in future conversations.

Do not use a fixed list of categories.

Judge each piece of information based on its long-term usefulness.

Remember information such as:
- Important personal context
- Preferences
- Relationships
- Projects
- Technical decisions
- Ongoing work
- Plans
- Goals
- Future events
- Important experiences
- Substantial context that may help the agent later

Do NOT remember information that is only temporary or useful
for the immediate moment, unless it has additional long-term value.

However, remember future events, meetings, appointments,
deadlines, plans, and scheduled activities because they may be
useful in future conversations even if they are time-sensitive.

Do not invent information.
Only extract information explicitly supported by the user's message.

For every memory, assign an importance score from 1 to 10.

For every memory, also assign a confidence score from 0 to 1. 
Confidence represent how certain the information is based on the  
user's wording. Explicit and definitive statements should recieve higher 
confidence, while uncertain, tentative or speculative statements 
should recieve lower confidence. 

Importance scale:

1-3:
Minor information with little future usefulness.

4-6:
Moderately useful information that may help in future conversations.

7-8:
Important information that provides meaningful long-term context.

9-10:
Highly important information that would substantially improve
future conversations or represents very important long-term context.

Prefer remembering useful information over aggressively filtering it.

For every memory, also provide a short reason explaining why the
information is worth remembering.

The reason must be based only on the user's message.

Return ONLY valid JSON in this exact format:

{{
    "memories": [
        {{
            "text": "standalone factual memory",
            "importance": 8,
            "confidence": 0.95,
            "reason": "why this information is worth remembering"
        }}
    ]
}}

If there is nothing worth remembering, return:

{{
    "memories": []
}}

User message:

{message}
"""

        started_at = time.perf_counter()
        extraction_started = time.perf_counter()
        try:
            with timed_operation("Gemini memory extraction"):
                extraction = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    )
                )

            response_text = extraction.text.strip()
            self._record_response_timing(
                "memory_extraction",
                extraction_started,
            )

            if not response_text:
                print("[MEMORY] Gemini returned an empty response.")
                return None

            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            result = json.loads(response_text)
            memories = result.get("memories", [])

            

            if not memories:
                self._record_memory_activity("save", "none")
                return None

            stored_any = False
            unique_memories = {}
            for memory in memories:
                if not isinstance(memory, dict):
                    continue
                memory_text = memory.get("text", "")
                if not isinstance(memory_text, str) or not memory_text.strip():
                    continue
                memory_text = memory_text.strip()
                key = self._normalized_memory_text(memory_text)
                if key not in unique_memories:
                    unique_memories[key] = {
                        **memory,
                        "text": memory_text,
                    }
                    continue
                existing = unique_memories[key]
                existing["importance"] = max(
                    existing.get("importance", 5),
                    memory.get("importance", 5),
                )
                existing["confidence"] = max(
                    existing.get("confidence", 0.5),
                    memory.get("confidence", 0.5),
                )

            for memory in unique_memories.values():
                memory_text = memory["text"]
                importance = memory.get("importance", 5)
                confidence = self._normalize_confidence(memory.get("confidence", 0.5))
                reason = memory.get("reason", "not specified")

                if not memory_text:
                    continue

                try:
                    pending_result = self._reconcile_matching_memory_operation(
                        memory_text,
                    )
                except Exception as error:
                    pending_result = "pending"
                    print(
                        "[MEMORY] Existing uncertain write could not yet be "
                        f"reconciled: {error}"
                    )
                if pending_result == "synchronized":
                    self._record_memory_activity("save", "merged", 1)
                    stored_any = True
                    continue
                if pending_result in {"pending", "untracked"}:
                    self._record_memory_activity(
                        "save",
                        "partial" if pending_result == "untracked" else "uncertain",
                        1,
                    )
                    if pending_result == "pending":
                        self._record_memory_outcome("outcome_uncertain")
                    stored_any = True
                    continue

                # Check whether this newly extracted memory duplicates
                # an existing memory before storing it.
                deduplication_started = time.perf_counter()
                try:
                    was_deduplicated = self.deduplicate_memory(
                        memory_text,
                        importance,
                        reason,
                        confidence,
                    )
                finally:
                    self._record_response_timing(
                        "memory_deduplication",
                        deduplication_started,
                    )

                if (
                    was_deduplicated
                    or getattr(self, "_last_deduplication_handled", False)
                ):
                    save_events = [
                        event
                        for event in self.last_response_metadata.get(
                            "memory_activity",
                            [],
                        )
                        if event["action"] == "save"
                    ]
                    if not save_events:
                        self._record_memory_activity("save", "merged", 1)
                        stored_any = True
                    elif any(
                        event["status"] in {
                            "uncertain",
                            "pending",
                            "partial",
                            "completed",
                        }
                        for event in save_events
                    ):
                        stored_any = True
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": memory_text
                    }
                ]

                write_result = self._write_memory_and_sync(
                    messages,
                    memory_text,
                    {
                        "importance": importance,
                        "confidence": confidence,
                        "reason": reason,
                        "source": "conversation"
                    },
                )

                if write_result["status"] in {
                    "saved_graph_synced",
                    "saved_graph_pending",
                    "saved_relationships_pending",
                    "saved_relationships_untracked",
                    "outcome_uncertain",
                    "succeeded_without_memory_id",
                    "timeout",
                }:
                    stored_any = True

            return stored_any

        except Exception as e:
            self._record_memory_activity("save", "failed")
            print(
                f"\n[ERROR] Memory extraction/storage failed: {e}"
            )
            traceback.print_exc()

            if "extraction" in locals():



             return None
        finally:
            self._record_response_timing("memory_save", started_at)

    def create_memory_relationships(self, memory_id, memory_text):
        """
        Find meaningful relationships between the new memory and existing memories
        in the user's knowledge graph. Store the relationships in the graph for
        future reference.
        """

        started_at = time.perf_counter()
        self._last_relationship_status = "untracked"
        try:
            if not self.graph.create_relationship_recovery(
                self.user_id,
                memory_id
            ):
                user_graph = self.graph.graph.get(self.user_id, {})
                queue = (
                    user_graph.get(self.graph.RELATIONSHIP_RECOVERY_KEY, {})
                    if isinstance(user_graph, dict)
                    else {}
                )
                if isinstance(queue, dict) and len(queue) >= (
                    self.graph.MAX_RECOVERY_RECORDS
                ):
                    reason = (
                        "the recovery queue is full; unresolved records "
                        "were preserved"
                    )
                else:
                    reason = "recovery tracking could not be recorded"
                print(
                    f"[GRAPH] {reason} for {memory_id}; generation was not "
                    "started."
                )
                return False

            self._last_relationship_status = "pending"
            relationships_synced = self._generate_tracked_relationships(
                memory_id,
                self.RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT
            )
            if relationships_synced:
                self._last_relationship_status = "synchronized"
            return relationships_synced
        except Exception as e:
            print(
                f"[GRAPH] Relationship recovery tracking failed for "
                f"{memory_id}; generation was not started: {e}"
            )
            return False
        finally:
            self._record_response_timing(
                "gemini_relationship_generation",
                started_at,
            )

    def _generate_relationship_edges(self, memory_id):
        user_graph = self.graph.get_user_graph(self.user_id)
        graph_nodes = user_graph.get("nodes", [])
        source = next(
            (
                node for node in graph_nodes
                if isinstance(node, dict) and node.get("id") == memory_id
            ),
            None
        )
        if source is None:
            raise ValueError("The source memory is not present in the user's graph.")

        existing_nodes = [
            node
            for node in graph_nodes
            if (
                isinstance(node, dict)
                and node.get("id") != memory_id
                and isinstance(node.get("id"), str)
                and node["id"].strip()
                and isinstance(node.get("memory_text"), str)
                and node["memory_text"].strip()
            )
        ]

        if not existing_nodes:
            return []

        candidate_nodes = [source] + existing_nodes
        memory_list = "\n".join(
            f"MEM0_ID: {node['id']} | MEMORY: {node['memory_text']}"
            for node in candidate_nodes
        )

        prompt = f"""
    You are a conservative knowledge-graph relationship analyzer.

    The supplied IDs are stable Mem0 memory IDs. Return them exactly;
    never invent IDs or use display numbers.

    USER-PARTITION MEMORIES:
    {memory_list}

    Find only precise, directed relationships explicitly supported by
    the memory text. Each relationship must be supported by evidence
    quoted from BOTH endpoint memories. You may connect two existing
    memories only when their direct relationship is stated by the
    supplied records. Do not infer a relationship from a shared topic,
    broad domain, model family, or mere co-occurrence.

    Rules:
    - Do not emit generic labels such as related_to, associated_with,
      about, or connected_to.
    - Use only these precise labels:
      has_backend, uses_backend, has_milestone, has_deadline,
      prerequisite_for, has_prerequisite, developed_on, runs_on,
      uses, implements, reports_performance_of,
      demonstrates_generalization_of, applies_to, part_of, member_of,
      located_in, studies, prefers, interested_in, motivated_by,
      describes, measured_on.
    - Direction is semantic. For a prerequisite chain, point from the
      prerequisite toward the dependent item and use prerequisite_for.
    - Return zero edges if evidence does not support a precise relation.
    - Include an exact quote from each endpoint. Evidence quotes must be
      substrings of the corresponding memory text.
    - Omit a relationship if either endpoint or either evidence quote is
      uncertain.

    Return ONLY valid JSON:

    {{
        "relationships": [
            {{
                "source_id": "source-mem0-id",
                "target_id": "target-mem0-id",
                "relationship": "precise_relationship_label",
                "evidence": [
                    {{"memory_id": "source-mem0-id", "quote": "exact source quote"}},
                    {{"memory_id": "target-mem0-id", "quote": "exact target quote"}}
                ]
            }}
        ]
    }}
    """

        with timed_operation("Gemini relationship generation"):
            response = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )

        result = json.loads(response.text)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("relationships"), list)
        ):
            raise ValueError(
                "Relationship response must contain a relationships list."
            )

        nodes_by_id = {
            node["id"]: node["memory_text"]
            for node in candidate_nodes
        }
        relationships = []
        seen = set()
        for item in result["relationships"]:
            if not isinstance(item, dict):
                raise ValueError("Relationship entries must be objects.")

            source_id = item.get("source_id")
            target_id = item.get("target_id")
            relationship = item.get("relationship")
            if (
                not isinstance(source_id, str)
                or source_id not in nodes_by_id
                or not isinstance(target_id, str)
                or not target_id.strip()
                or target_id not in nodes_by_id
                or source_id == target_id
                or not isinstance(relationship, str)
                or relationship not in self.RELATIONSHIP_LABELS
                or not isinstance(item.get("evidence"), list)
                or len(item["evidence"]) < 2
                or len(item["evidence"]) > 5
            ):
                raise ValueError(
                    "Relationship entry contains invalid endpoints or label."
                )
            evidence_ids = []
            for evidence in item["evidence"]:
                if (
                    not isinstance(evidence, dict)
                    or evidence.get("memory_id") not in nodes_by_id
                    or not isinstance(evidence.get("quote"), str)
                    or not evidence["quote"].strip()
                    or len(evidence["quote"]) > 500
                    or evidence["quote"]
                    not in nodes_by_id[evidence["memory_id"]]
                ):
                    raise ValueError(
                        "Relationship evidence must quote existing memory text."
                    )
                evidence_ids.append(evidence["memory_id"])
            if source_id not in evidence_ids or target_id not in evidence_ids:
                raise ValueError(
                    "Relationship evidence must cite both endpoint memories."
                )
            identity = (source_id, target_id, relationship)
            if identity in seen:
                continue
            seen.add(identity)
            edge = {
                "source_id": source_id,
                "target_id": target_id,
                "relationship": relationship,
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            }
            relationships.append(edge)

        if len(relationships) > self.graph.MAX_EXPECTED_EDGES:
            raise ValueError("Relationship response exceeds the edge limit.")
        return relationships

    def _generate_tracked_relationships(
        self,
        memory_id,
        attempt_limit,
        expected_user_graph=None,
        on_attempt_started=None,
    ):
        attempt = self.graph.begin_relationship_attempt(
            self.user_id,
            memory_id,
            attempt_limit,
            expected_user_graph=expected_user_graph,
        )
        if attempt["status"] != "ready":
            print(
                f"[GRAPH] Relationship generation for {memory_id} was not "
                f"started: {attempt['status']}."
            )
            return False
        if on_attempt_started is not None:
            on_attempt_started()

        try:
            edges = self._generate_relationship_edges(memory_id)
        except Exception as e:
            try:
                self.graph.set_relationship_recovery_status(
                    self.user_id,
                    memory_id,
                    "failed",
                    str(e)
                )
            except Exception as save_error:
                print(
                    f"[GRAPH] Generation failed for {memory_id}, and failure "
                    f"status could not be saved: {save_error}"
                )
            print(
                f"[GRAPH] Relationship generation failed for {memory_id}: {e}"
            )
            return False

        if not edges:
            try:
                if not self.graph.finish_empty_relationship_recovery(
                    self.user_id,
                    memory_id
                ):
                    raise ValueError("Could not confirm empty-result status.")
                return True
            except Exception as e:
                print(
                    f"[GRAPH] Empty relationship result for {memory_id} was "
                    f"not durably recorded: {e}"
                )
                return False

        try:
            if not self.graph.set_pending_relationship_edges(
                self.user_id,
                memory_id,
                edges
            ):
                raise ValueError("Could not persist expected relationship edges.")
            result = self.graph.complete_relationship_recovery(
                self.user_id,
                memory_id
            )
            if result["status"] == "blocked_missing_endpoint":
                self.graph.set_relationship_recovery_status(
                    self.user_id,
                    memory_id,
                    "blocked_missing_endpoint",
                    "A source or target node no longer exists."
                )
                return False
            return result["status"] == "succeeded"
        except Exception as e:
            print(
                f"[GRAPH] Relationship persistence remains pending for "
                f"{memory_id}: {e}"
            )
            return False

    def _relationship_recovery_snapshot(self):
        with self.graph._lock:
            user_graph = copy.deepcopy(
                self.graph.graph.get(self.user_id, {})
            )
            recovery = self.graph.get_relationship_recovery(
                self.user_id,
                user_graph=user_graph,
            )
        return user_graph, recovery

    def _evaluate_relationship_recovery(
        self,
        user_graph,
        recovery,
        source_filter,
        attempt_limit,
    ):
        return evaluate_relationship_recovery(
            user_graph,
            recovery,
            source_filter,
            attempt_limit,
        )

    def recover_relationships(
        self,
        dry_run=True,
        batch_limit=RELATIONSHIP_RECOVERY_BATCH_LIMIT,
        attempt_limit=RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT,
        source_id=None,
    ):
        if type(batch_limit) is not int or batch_limit < 1:
            raise ValueError("batch_limit must be a positive integer.")
        if type(attempt_limit) is not int or attempt_limit < 1:
            raise ValueError("attempt_limit must be a positive integer.")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string.")
        source_filter = source_id
        batch_limit = min(batch_limit, self.RELATIONSHIP_RECOVERY_BATCH_LIMIT)
        attempt_limit = min(
            attempt_limit,
            self.RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT,
            self.graph.MAX_RECOVERY_ATTEMPTS,
        )
        user_graph, recovery = self._relationship_recovery_snapshot()
        report = {
            "dry_run": dry_run,
            "batch_limit": batch_limit,
            "attempted": 0,
            "succeeded": 0,
            "succeeded_empty": 0,
            "pending": 0,
            "failed": 0,
            "blocked": 0,
            "capped": [],
            "eligible": [],
            "invalid_records": recovery["invalid_records"],
            "queue_size": recovery["queue_size"],
            "remaining": 0,
            "details": [],
        }

        if not recovery["valid"]:
            report["error"] = recovery["error"]
            report["blocked"] = recovery["blocked_count"]
            return report
        evaluation = self._evaluate_relationship_recovery(
            user_graph,
            recovery,
            source_filter,
            attempt_limit,
        )
        report["invalid_records"] = evaluation["invalid_records"]
        eligible = evaluation["eligible"]
        report["eligible"] = [record["source_id"] for record in eligible]
        report["details"].extend(evaluation["details"])
        report["capped"] = evaluation["capped"]
        report["blocked"] += evaluation["blocked"]
        if source_filter is not None:
            report["blocked"] += len(report["invalid_records"])

        if not dry_run:
            for detail in evaluation["details"]:
                if detail["status"] != "blocked_missing_endpoint":
                    continue
                source_to_block = detail["source_id"]
                try:
                    self.graph.set_relationship_recovery_status(
                        self.user_id,
                        source_to_block,
                        "blocked_missing_endpoint",
                        detail.get("reason") or "A source or target node is missing.",
                    )
                except Exception as error:
                    detail["reason"] = str(error)

        if source_filter is not None and not any(
            record["source_id"] == source_filter
            for record in recovery["records"]
        ) and source_filter not in report["invalid_records"]:
            report["details"].append(
                {
                    "source_id": source_filter,
                    "status": "recovery_record_not_found",
                }
            )
        report["remaining"] = max(0, len(eligible) - batch_limit)
        if dry_run:
            report["pending"] = len(eligible)
            if source_filter is None:
                report["blocked"] += len(report["invalid_records"])
            return report

        for record in eligible[:batch_limit]:
            source_id = record["source_id"]
            try:
                current_user_graph, current_recovery = (
                    self._relationship_recovery_snapshot()
                )
                current_evaluation = self._evaluate_relationship_recovery(
                    current_user_graph,
                    current_recovery,
                    source_id,
                    attempt_limit,
                )
                current_record = next(
                    (
                        candidate
                        for candidate in current_evaluation["eligible"]
                        if candidate["source_id"] == source_id
                    ),
                    None,
                )
                if (
                    source_filter is not None
                    and current_user_graph != user_graph
                ):
                    report["details"].append(
                        {
                            "source_id": source_id,
                            "status": "state_changed",
                            "reason": (
                                "Recovery state changed after eligibility "
                                "was evaluated."
                            ),
                        }
                    )
                    continue
                if current_record is None:
                    report["blocked"] += current_evaluation["blocked"]
                    report["capped"].extend(current_evaluation["capped"])
                    report["details"].extend(current_evaluation["details"])
                    if not current_recovery["valid"]:
                        report["error"] = current_recovery["error"]
                    elif not any(
                        detail.get("source_id") == source_id
                        for detail in current_evaluation["details"]
                    ):
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "state_changed",
                                "reason": (
                                    "Recovery record is no longer eligible."
                                ),
                            }
                        )
                    continue

                status = current_record["status"]
                if status == "pending_persistence":
                    result = self.graph.complete_relationship_recovery(
                        self.user_id,
                        source_id,
                        expected_user_graph=current_user_graph,
                    )
                    report["attempted"] += 1
                    if result["status"] == "succeeded":
                        report["succeeded"] += 1
                        report["details"].append(
                            {"source_id": source_id, "status": "succeeded"}
                        )
                    elif result["status"] == "blocked_missing_endpoint":
                        reason = "A source or target node is missing."
                        self.graph.set_relationship_recovery_status(
                            self.user_id,
                            source_id,
                            "blocked_missing_endpoint",
                            reason
                        )
                        report["blocked"] += 1
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "blocked",
                                "reason": reason,
                            }
                        )
                    else:
                        report["pending"] += 1
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "pending",
                                "reason": "Expected edges could not be persisted.",
                            }
                        )
                else:
                    was_successful = self._generate_tracked_relationships(
                        source_id,
                        attempt_limit,
                        expected_user_graph=current_user_graph,
                        on_attempt_started=lambda: report.__setitem__(
                            "attempted",
                            report["attempted"] + 1,
                        ),
                    )
                    updated = self.graph.get_relationship_recovery(
                        self.user_id
                    )
                    refreshed = next(
                        (
                            item for item in updated["records"]
                            if item["source_id"] == source_id
                        ),
                        None
                    )
                    if was_successful and refreshed:
                        if refreshed["status"] == "succeeded_empty":
                            report["succeeded_empty"] += 1
                            report["details"].append(
                                {
                                    "source_id": source_id,
                                    "status": "succeeded-empty",
                                }
                            )
                        else:
                            report["succeeded"] += 1
                            report["details"].append(
                                {"source_id": source_id, "status": "succeeded"}
                            )
                    elif refreshed and refreshed["status"] == "failed":
                        report["failed"] += 1
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "failed",
                                "reason": refreshed.get("error") or
                                    "Relationship generation failed.",
                            }
                        )
                    elif (
                        refreshed
                        and refreshed["status"] == "blocked_missing_endpoint"
                    ):
                        report["blocked"] += 1
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "blocked",
                                "reason": refreshed.get("error") or
                                    "A source or target node is missing.",
                            }
                        )
                    else:
                        report["pending"] += 1
                        report["details"].append(
                            {
                                "source_id": source_id,
                                "status": "pending",
                                "reason": (
                                    refreshed.get("error")
                                    if refreshed
                                    else "Recovery status could not be confirmed."
                                ),
                            }
                        )
            except self.graph.StateConflictError as e:
                report["details"].append(
                    {
                        "source_id": source_id,
                        "status": "state_changed",
                        "reason": str(e),
                    }
                )
            except Exception as e:
                report["pending"] += 1
                report["details"].append(
                    {
                        "source_id": source_id,
                        "status": "pending",
                        "reason": str(e)[:self.graph.MAX_RECOVERY_ERROR_LENGTH],
                    }
                )

        report["pending"] += report["remaining"]
        if source_filter is None:
            report["blocked"] += recovery["blocked_count"]
        return report

    def enroll_relationship_recovery(self, source_id):
        if not isinstance(source_id, str) or not source_id.strip():
            return {"status": "invalid_source"}

        recovery = self.graph.get_relationship_recovery(self.user_id)
        if not recovery["valid"]:
            return {
                "status": "invalid_queue",
                "reason": recovery["error"],
            }
        if recovery["invalid_records"]:
            return {
                "status": "invalid_queue",
                "reason": "The recovery queue contains invalid records.",
            }

        user_graph = self.graph.graph.get(self.user_id)
        if not isinstance(user_graph, dict):
            return {"status": "missing_node"}
        nodes = user_graph.get("nodes")
        if not isinstance(nodes, list):
            return {"status": "invalid_graph"}

        source_node = next(
            (
                node for node in nodes
                if isinstance(node, dict) and node.get("id") == source_id
            ),
            None,
        )
        if source_node is None:
            return {"status": "missing_node"}
        memory_text = source_node.get("memory_text")
        if not isinstance(memory_text, str) or not memory_text.strip():
            return {"status": "missing_text"}

        existing_record = next(
            (
                record for record in recovery["records"]
                if record["source_id"] == source_id
            ),
            None,
        )
        if existing_record is not None:
            return {
                "status": "already_tracked",
                "record_status": existing_record["status"],
            }

        if recovery["queue_size"] >= self.graph.MAX_RECOVERY_RECORDS:
            return {"status": "queue_full"}

        if not self.graph.create_relationship_recovery(
            self.user_id,
            source_id,
        ):
            return {"status": "enrollment_failed"}

        refreshed = self.graph.get_relationship_recovery(self.user_id)
        record = next(
            (
                item for item in refreshed["records"]
                if item["source_id"] == source_id
            ),
            None,
        )
        if record is None:
            return {"status": "enrollment_failed"}
        if record["status"] != "pending_generation" or record["attempts"] != 0:
            return {
                "status": "already_tracked",
                "record_status": record["status"],
            }

        return {"status": "enrolled"}

    def inspect_relationship_recovery(self, source_id):
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a non-empty string.")

        user_graph, recovery = self._relationship_recovery_snapshot()
        nodes = (
            user_graph.get("nodes", [])
            if isinstance(user_graph, dict)
            else []
        )
        if not isinstance(nodes, list):
            nodes = []
        node_ids = {
            node.get("id")
            for node in nodes
            if (
                isinstance(node, dict)
                and isinstance(node.get("id"), str)
                and node["id"].strip()
            )
        }
        node_exists = source_id in node_ids

        record_evaluation = self._evaluate_relationship_recovery(
            user_graph,
            recovery,
            source_id,
            min(
                self.RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT,
                self.graph.MAX_RECOVERY_ATTEMPTS,
            ),
        )
        record = next(
            (
                item for item in recovery["records"]
                if item["source_id"] == source_id
            ),
            None,
        )
        invalid_record = source_id in record_evaluation["invalid_records"]
        attempt_cap = min(
            self.RELATIONSHIP_RECOVERY_ATTEMPT_LIMIT,
            self.graph.MAX_RECOVERY_ATTEMPTS,
        )
        record_status = record["status"] if record else None
        attempts = record["attempts"] if record else None
        cap_reached = bool(
            record
            and record_status in {"pending_generation", "failed"}
            and attempts >= attempt_cap
        )
        eligibility = {
            "error": recovery["error"] if not recovery["valid"] else None,
            "eligible": [
                item["source_id"]
                for item in record_evaluation["eligible"]
            ],
            "capped": record_evaluation["capped"],
            "details": record_evaluation["details"],
        }
        eligible = source_id in eligibility["eligible"]
        reason = self._relationship_eligibility_reason(
            source_id,
            eligibility,
            record,
            invalid_record,
            node_exists,
        )

        return {
            "node_exists": node_exists,
            "record_exists": record is not None,
            "record_valid": record is not None,
            "invalid_record": invalid_record,
            "queue_valid": recovery["valid"],
            "status": record_status,
            "attempts": attempts,
            "attempt_cap": attempt_cap,
            "attempt_cap_reached": cap_reached,
            "eligible": eligible,
            "reason": reason,
            "node_count": len(nodes),
            "recovery_record_count": recovery["queue_size"],
            "valid_recovery_record_count": len(recovery["records"]),
        }

    @staticmethod
    def _relationship_eligibility_reason(
        source_id,
        report,
        record=None,
        invalid_record=False,
        node_exists=True,
    ):
        if report.get("error"):
            return "invalid_recovery_queue"
        if source_id in report.get("eligible", []):
            return "eligible"
        if source_id in report.get("capped", []):
            return "attempt_cap_reached"
        if invalid_record:
            return "invalid_recovery_record"
        if record is None:
            return "recovery_record_not_found"
        if record["status"] in {"succeeded", "succeeded_empty", "superseded"}:
            return "record_not_eligible"
        if not node_exists:
            return "source_node_missing"
        for detail in report.get("details", []):
            if detail.get("source_id") != source_id:
                continue
            if detail.get("missing_target_ids"):
                return "target_node_missing"
            if detail.get("status") == "blocked_missing_endpoint":
                return "blocked_missing_endpoint"
            if detail.get("status") == "blocked_invalid_record":
                return "blocked_invalid_record"
        return "not_eligible"

    @staticmethod
    def _diagnostic_fingerprint(snapshot):
        try:
            canonical = json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(canonical).hexdigest()[:12]

    def relationship_runtime_diagnostics(
        self,
        source_id=None,
        eligibility_report=None,
    ):
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string.")
        if eligibility_report is None:
            eligibility_report = self.recover_relationships(
                dry_run=True,
                source_id=source_id,
            )

        with self.graph._lock:
            active_snapshot = copy.deepcopy(
                self.graph.graph.get(self.user_id)
            )
            graph_path = os.path.realpath(
                os.path.abspath(self.graph.path_file)
            )

        active_user_graph = (
            active_snapshot
            if isinstance(active_snapshot, dict)
            else {}
        )
        nodes = active_user_graph.get("nodes", [])
        relationships = active_user_graph.get("relationships", [])
        recovery_queue = active_user_graph.get(
            self.graph.RELATIONSHIP_RECOVERY_KEY,
            {},
        )
        nodes = nodes if isinstance(nodes, list) else []
        relationships = relationships if isinstance(relationships, list) else []
        recovery_queue = (
            recovery_queue if isinstance(recovery_queue, dict) else {}
        )

        valid_records = [
            record
            for key, record in recovery_queue.items()
            if (
                self.graph._valid_recovery_record(record)
                and key == record.get("source_id")
            )
        ]
        invalid_record = any(
            key == source_id
            and (
                not self.graph._valid_recovery_record(record)
                or key != record.get("source_id")
            )
            for key, record in recovery_queue.items()
        )
        record = next(
            (
                item for item in valid_records
                if item["source_id"] == source_id
            ),
            None,
        ) if source_id is not None else None
        node_exists = any(
            isinstance(node, dict) and node.get("id") == source_id
            for node in nodes
        ) if source_id is not None else None
        reason = (
            self._relationship_eligibility_reason(
                source_id,
                eligibility_report,
                record,
                invalid_record,
                node_exists,
            )
            if source_id is not None
            else "not_requested"
        )

        disk_fingerprint = None
        disk_error = None
        disk_counts = {
            "nodes": None,
            "relationships": None,
            "recovery_records": None,
            "valid_recovery_records": None,
        }
        try:
            with open(graph_path, "r", encoding="utf-8") as graph_file:
                persisted_graph = json.load(graph_file)
            if not isinstance(persisted_graph, dict):
                disk_error = "invalid_graph_root"
            else:
                persisted_user_graph = persisted_graph.get(self.user_id)
                if persisted_user_graph is None:
                    persisted_user_graph = {}
                if not isinstance(persisted_user_graph, dict):
                    disk_error = "invalid_user_graph"
                else:
                    disk_fingerprint = self._diagnostic_fingerprint(
                        persisted_user_graph
                    )
                    disk_nodes = persisted_user_graph.get("nodes", [])
                    disk_relationships = persisted_user_graph.get(
                        "relationships",
                        [],
                    )
                    disk_recovery = persisted_user_graph.get(
                        self.graph.RELATIONSHIP_RECOVERY_KEY,
                        {},
                    )
                    if (
                        not isinstance(disk_nodes, list)
                        or not isinstance(disk_relationships, list)
                        or not isinstance(disk_recovery, dict)
                    ):
                        disk_error = "invalid_graph_structure"
                    disk_nodes = (
                        disk_nodes if isinstance(disk_nodes, list) else []
                    )
                    disk_relationships = (
                        disk_relationships
                        if isinstance(disk_relationships, list)
                        else []
                    )
                    disk_recovery = (
                        disk_recovery
                        if isinstance(disk_recovery, dict)
                        else {}
                    )
                    disk_counts = {
                        "nodes": len(disk_nodes),
                        "relationships": len(disk_relationships),
                        "recovery_records": len(disk_recovery),
                        "valid_recovery_records": sum(
                            1
                            for key, item in disk_recovery.items()
                            if (
                                self.graph._valid_recovery_record(item)
                                and key == item.get("source_id")
                            )
                        ),
                    }
        except FileNotFoundError:
            disk_error = "file_not_found"
        except PermissionError:
            disk_error = "permission_denied"
        except json.JSONDecodeError:
            disk_error = "invalid_json"
        except UnicodeDecodeError:
            disk_error = "invalid_encoding"
        except OSError:
            disk_error = "read_error"

        return {
            "process_id": os.getpid(),
            "graph_path": graph_path,
            "memory_fingerprint": self._diagnostic_fingerprint(
                active_snapshot
            ),
            "disk_fingerprint": disk_fingerprint,
            "disk_error": disk_error,
            "node_count": len(nodes),
            "relationship_count": len(relationships),
            "recovery_record_count": len(recovery_queue),
            "valid_recovery_record_count": len(valid_records),
            "disk_node_count": disk_counts["nodes"],
            "disk_relationship_count": disk_counts["relationships"],
            "disk_recovery_record_count": disk_counts["recovery_records"],
            "disk_valid_recovery_record_count": (
                disk_counts["valid_recovery_records"]
            ),
            "source_node_exists": node_exists,
            "source_record_exists": record is not None,
            "source_record_valid": record is not None,
            "source_record_status": record["status"] if record else None,
            "source_record_attempts": record["attempts"] if record else None,
            "source_eligible": (
                source_id in eligibility_report.get("eligible", [])
                if source_id is not None
                else None
            ),
            "source_eligibility_reason": reason,
            "invalid_source_record": invalid_record,
        }

    def verify_remote_memory(self, memory_id):
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string.")

        try:
            listing = self.memory.get_all_complete(self.user_id)
        except Exception:
            return {
                "found": None,
                "listing_complete": False,
                "listing_valid": False,
                "memory_count": None,
                "result_count": None,
                "result": "inconclusive",
            }

        if (
            not isinstance(listing, dict)
            or type(listing.get("count")) is not int
            or listing["count"] < 0
            or not isinstance(listing.get("results"), list)
            or listing["count"] != len(listing["results"])
        ):
            return {
                "found": None,
                "listing_complete": False,
                "listing_valid": False,
                "memory_count": (
                    listing.get("count")
                    if isinstance(listing, dict)
                    and type(listing.get("count")) is int
                    and listing["count"] >= 0
                    else None
                ),
                "result_count": (
                    len(listing["results"])
                    if isinstance(listing, dict)
                    and isinstance(listing.get("results"), list)
                    else None
                ),
                "result": "inconclusive",
            }

        memories = listing["results"]
        memory_ids = []
        for memory in memories:
            if (
                not isinstance(memory, dict)
                or not isinstance(memory.get("id"), str)
                or not memory["id"].strip()
                or not isinstance(memory.get("memory"), str)
                or not memory["memory"].strip()
            ):
                return {
                    "found": None,
                    "listing_complete": False,
                    "listing_valid": False,
                    "memory_count": listing["count"],
                    "result_count": len(memories),
                    "result": "inconclusive",
                }
            memory_ids.append(memory["id"])

        if len(memory_ids) != len(set(memory_ids)):
            return {
                "found": None,
                "listing_complete": False,
                "listing_valid": False,
                "memory_count": listing["count"],
                "result_count": len(memories),
                "result": "inconclusive",
            }

        found = memory_id in set(memory_ids)
        return {
            "found": found,
            "listing_complete": True,
            "listing_valid": True,
            "memory_count": listing["count"],
            "result_count": len(memories),
            "result": "found" if found else "absent",
        }

    def restore_local_recovery_source(self, replacement_id):
        if not isinstance(replacement_id, str) or not replacement_id.strip():
            return {"status": "invalid_input"}

        try:
            listing = self.memory.get_all_complete(self.user_id)
        except Exception:
            return {"status": "listing_inconclusive"}

        if (
            not isinstance(listing, dict)
            or type(listing.get("count")) is not int
            or listing["count"] < 0
            or not isinstance(listing.get("results"), list)
            or listing["count"] != len(listing["results"])
        ):
            return {"status": "listing_inconclusive"}

        memories = listing["results"]
        memory_ids = []
        requested_matches = []
        for memory in memories:
            if (
                not isinstance(memory, dict)
                or not isinstance(memory.get("id"), str)
                or not memory["id"].strip()
                or not isinstance(memory.get("memory"), str)
                or not memory["memory"].strip()
            ):
                return {"status": "listing_inconclusive"}
            memory_ids.append(memory["id"])
            if memory["id"] == replacement_id and memory["memory"].strip():
                requested_matches.append(memory)

        if len(memory_ids) != len(set(memory_ids)):
            return {"status": "listing_inconclusive"}
        if not requested_matches:
            return {
                "status": "remote_source_not_found",
                "listing_count": listing["count"],
            }
        if len(requested_matches) != 1:
            return {
                "status": "remote_source_ambiguous",
                "listing_count": listing["count"],
            }

        result = self.graph.restore_local_recovery_source(
            self.user_id,
            replacement_id,
            requested_matches[0]["memory"],
        )
        result["listing_count"] = listing["count"]
        return result

    def deduplicate_memory(self, memory_text, importance, reason, confidence):
        """
        Check whether a newly extracted memory is a true duplicate
        of an existing memory.

        Memories that are merely related or about the same topic should
        remain separate if they contain different information.
        """

        self._last_deduplication_handled = False
        try:
            # Find potentially related memories using semantic search.
            similar_memories = self.recall(memory_text)

            if not similar_memories or "results" not in similar_memories:
                return False

            # Only send reasonably relevant memories to Gemini.
            candidates = [
                item
                for item in similar_memories["results"]
                if item.get("score", 0) >= 0.35
            ]

            candidates = sorted(
                candidates,
                key=lambda item: item.get("score", 0),
                reverse=True
            )[:5]

            if not candidates:
                return False

            memory_list = "\n".join(
                f"ID: {memory['id']} | Memory: {memory['memory']}"
                f"Confidence: {memory.get('metadata', {}).get('confidence', 'N/A')}"
                for memory in candidates
            )

            prompt = f"""
You are the memory deduplication system for an AI assistant.

Your job is to determine whether the NEW MEMORY is a true duplicate
of any EXISTING MEMORY.

NEW MEMORY:
"{memory_text}"

EXISTING MEMORIES:
{memory_list}

IMPORTANT DEFINITION:

A duplicate means BOTH memories communicate the SAME underlying user fact.

Being about the same topic, subject, person, technology, object,
or preference does NOT make two memories duplicates.

If the new memory contains ANY meaningful information that is not
already contained in the existing memory, it is NOT a duplicate.

In particular, differences in any of the following mean the memories
should normally remain separate:

- usage
- role
- context
- reason
- relationship
- location
- time
- quantity
- project
- activity
- skill
- experience
- additional preference
- any other meaningful detail

DO NOT merge memories merely because they mention the same thing.

CRITICAL EXAMPLES:

Existing:
"User likes Python."

New:
"User uses Python for backend development."

Result:
NOT DUPLICATE.

Reason:
The first memory describes a preference.
The second describes how the user uses Python.
They contain different facts.

---

Existing:
"User likes Python."

New:
"Python is the user's favorite programming language."

Result:
DUPLICATE.

Reason:
Both express the same preference.

---

Existing:
"User prefers dark mode."

New:
"User prefers dark themes in applications."

Result:
DUPLICATE.

Reason:
Both express the same preference.

---

Existing:
"User likes apples."

New:
"User likes mangoes."

Result:
NOT DUPLICATE.

Reason:
They are different preferences.

---

Existing:
"User works at Google."

New:
"User is employed by Google."

Result:
DUPLICATE.

Reason:
Both express the same employment fact.

---

Existing:
"User works at Google."

New:
"User works as a software engineer at Google."

Result:
NOT DUPLICATE.

Reason:
The second memory adds a job role.

---

DECISION RULE:

Ask:

"Does the new memory contain any meaningful fact that the existing
memory does not already contain?"

If YES:
→ NOT DUPLICATE.

If NO, and both memories express the same underlying fact:
→ DUPLICATE.

When in doubt, DO NOT merge. Preserve the information.

If the new memory is not a duplicate, return an empty list.

If duplicates exist, return the IDs of ONLY the true duplicates.

If duplicates exist, create a merged memory that preserves the
same underlying fact without inventing information.

Return ONLY valid JSON.

Format when duplicate exists:

{{
    "duplicate_memory_ids": ["existing-memory-id"],
    "merged_memory": "complete standalone memory",
    "importance": 8,
    "confidence": 0.95,
    "reason": "why this memory is useful"
}}

Format when there is no duplicate:

{{
    "duplicate_memory_ids": [],
    "merged_memory": "",
    "importance": 0,
    "reason": ""
}}
"""
            
            with timed_operation("Gemini memory deduplication"):
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    )
                )

            response_text = response.text.strip()

            if not response_text:
                print(
                    "[MEMORY] Gemini returned an empty "
                    "deduplication response."
                )
                return False

            # Remove Markdown code fences if Gemini ever returns them.
            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            result = json.loads(response_text)

            duplicate_ids = result.get(
                "duplicate_memory_ids",
                []
            )

            if not duplicate_ids:
                return False

            # Only allow Gemini to select memories that our application
            # actually provided as candidates.
            valid_ids = {
                memory["id"]
                for memory in candidates
            }

            duplicate_ids = [
                memory_id
                for memory_id in duplicate_ids
                if memory_id in valid_ids
            ]
            duplicate_ids = list(dict.fromkeys(duplicate_ids))

            if not duplicate_ids:
                return False

            merged_memory = result.get(
                "merged_memory",
                ""
            ).strip()

            if not merged_memory:
                print(
                    "[MEMORY] Deduplication found duplicates "
                    "but no merged memory was returned."
                )
                return False

            merged_importance = result.get(
                "importance",
                importance
            )

            merged_confidence = self._normalize_confidence(
                result.get("confidence", confidence), 
                confidence
            )

            merged_reason = result.get(
                "reason",
                "Merged from duplicate memories."
            )
            candidate_by_id = {
                memory["id"]: memory
                for memory in candidates
            }
            source_memories = [
                candidate_by_id[memory_id]
                for memory_id in duplicate_ids
            ]
            merged_metadata = self._merge_deduplication_metadata(
                source_memories,
                merged_importance,
                merged_confidence,
                merged_reason
            )
            merged_metadata["source"] = source_memories[0].get(
                "metadata", {}
            ).get("source", "conversation") if isinstance(
                source_memories[0].get("metadata", {}),
                dict
            ) else "conversation"

            # Create and confirm the replacement before deleting originals.
            self._last_deduplication_handled = True
            messages = [
                {
                    "role": "user",
                    "content": merged_memory
                }
            ]

            write_result = self._write_memory_and_sync(
                messages,
                merged_memory,
                merged_metadata,
                operation_type="merge",
                related_memory_ids=duplicate_ids,
                create_relationships=False,
            )

            merged_memory_id = write_result.get("memory_id")
            if write_result["status"] != "saved_graph_synced":
                print(
                    "[MEMORY] Merged memory was not fully synchronized; "
                    "original memories were preserved."
                )
                return False

            deleted_ids = []
            failed_deletions = []
            for duplicate_id in duplicate_ids:
                try:
                    delete_result = self.memory.delete(duplicate_id)
                    if self._delete_succeeded(delete_result):
                        deleted_ids.append(duplicate_id)
                    else:
                        failed_deletions.append(duplicate_id)
                        print(
                            f"[MEMORY] Could not confirm deletion of "
                            f"duplicate memory {duplicate_id}."
                        )
                except Exception as e:
                    failed_deletions.append(duplicate_id)
                    print(
                        f"[MEMORY] Could not delete duplicate memory "
                        f"{duplicate_id}: {e}"
                    )

            graph_errors = []
            graph_started = time.perf_counter()
            for duplicate_id in deleted_ids:
                try:
                    self.graph.delete_node(
                        self.user_id,
                        duplicate_id
                    )
                except Exception as e:
                    graph_errors.append(str(e))
            self._record_response_timing(
                "graph_persistence",
                graph_started,
            )

            graph_started = time.perf_counter()
            try:
                self.graph.add_node(
                    self.user_id,
                    merged_memory_id,
                    merged_memory
                )
            except Exception as e:
                graph_errors.append(str(e))
            finally:
                self._record_response_timing(
                    "graph_persistence",
                    graph_started,
                )
            try:
                if not self.create_memory_relationships(
                    merged_memory_id,
                    merged_memory
                ):
                    graph_errors.append("Relationship creation failed.")
            except Exception as e:
                graph_errors.append(str(e))

            if failed_deletions or graph_errors:
                print(
                    "[MEMORY] Merge partially completed. "
                    f"Merged memory ID: {merged_memory_id}; "
                    f"duplicates not confirmed deleted: {failed_deletions}; "
                    f"graph reconciliation errors: {graph_errors}"
                )
                return True

            print(
                "[MEMORY] Duplicate memories merged successfully."
            )

            return True

        except Exception as e:
            print(
                f"\n[ERROR] Memory deduplication failed: {e}\n"
            )

            if "response" in locals():
                

            # If deduplication fails, allow the caller to store
            # the new memory normally.
             return False

    def recall(self, query):
        """
        Search the user's long-term memories.
        """

        started_at = time.perf_counter()
        try:
            results = self.memory.search(
                query,
                self.user_id
            )
            memories = (
                results.get("results", [])
                if isinstance(results, dict)
                and isinstance(results.get("results"), list)
                else []
            )
            self._record_memory_activity(
                "retrieval",
                "completed" if memories else "none",
                len(memories),
            )
            return results

        except Exception as e:
            self._record_memory_activity("retrieval", "failed")
            print(
                f"\n[ERROR] Memory search failed: {e}\n"
            )
            raise
        finally:
            self._record_response_timing("memory_retrieval", started_at)

    def update_memory(self, message, candidate_memories):
        """
        Use Gemini to determine whether the new message changes any
        relevant existing memories and, if so, create the updated versions.

        This runs on every user message. It does not depend on keywords.
        """

        self._last_update_handled = False
        if not candidate_memories:
            return False

        started_at = time.perf_counter()
        try:
            memory_list = "\n".join(
                f"ID: {memory['id']} | Memory: {memory['memory']}"
                f"Confidence: {memory.get('metadata', {}).get('confidence', 'N/A')}"
                for memory in candidate_memories
            )

            prompt = f"""
You are the memory evolution system for an AI assistant.

Your job is to determine whether the user's NEW message changes,
updates, replaces, or corrects any of the EXISTING memories below.

Do this using semantic understanding, not keywords.
A user may express a change indirectly, briefly, or using a reference
such as "that meeting", "make it 12", or "I prefer the other one".
Use the existing memories and the new message together to understand
what the user means.

Existing relevant memories:

{memory_list}

New user message:

"{message}"

Rules:
1. Return an update only when the new message clearly changes,
   corrects, replaces, or modifies an existing memory.
2. Do not treat a merely related message as an update.

IMPORTANT:
A new fact that adds a different detail about the same topic is NOT an update.

For example:

Existing:
"User enjoys photography."

New:
"User uses Lightroom to edit travel photos."

Result:
NO UPDATE.

Reason:
The new message adds information about a tool and activity.
It does not change whether the user enjoys photography.

Only update an existing memory when the new message changes
the actual fact expressed by that memory.

3. Resolve references such as "that", "it", "the meeting", or "the old one"
   using the existing memories when the meaning is clear.
4. When a memory is updated, rewrite it as a complete standalone factual
   statement containing the newest information.
5. Preserve important information from the old memory that is still true.
6. Do not invent facts or details that are not supported by the old memory
   and the new message.
7. If multiple memories are changed, return all of them.
8. If nothing is changed, return an empty updates list.
9. For every updated memory, provide an importance score from 1 to 10
   and a short reason based on the user's new message and the updated fact.
10. For every updated memory, also provide a confidence score from 0 to 1
    based on how certain the updated fact is supported by the user's message. 

Return ONLY valid JSON in this exact format:

{{
    "updates": [
        {{
            "memory_id": "existing-memory-id",
            "text": "complete updated standalone memory",
            "importance": 8,
            "reason": "why the updated information is worth remembering",
            "confidence": 0.95
        }}
    ]
}}

If there are no updates, return:

{{
    "updates": []
}}
"""
            with timed_operation("Gemini memory update analysis"):
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    )
                )

            response_text = response.text.strip()

            if not response_text:
                print("[MEMORY] Gemini returned an empty update response.")
                return False

            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            result = json.loads(response_text)
            updates = result.get("updates", [])

            if not updates:
                return False

            valid_ids = {memory["id"] for memory in candidate_memories}
            updated_any = False

            for update in updates:
                memory_id = update.get("memory_id")

                old_memory = next(
                    (
                        memory
                        for memory in candidate_memories
                        if memory["id"] == memory_id
                    ),
                    None
                )

                old_memory_text = (
                    old_memory.get("memory", "")
                    if old_memory
                    else ""
                )

                old_metadata = old_memory.get("metadata", {}) if old_memory else {}
                if not isinstance(old_metadata, dict):
                    old_metadata = {}

                memory_text = update.get("text", "").strip()
                importance = update.get(
                    "importance",
                    old_metadata.get("importance", 5)
                )
                confidence = self._normalize_confidence(
                    update.get(
                        "confidence",
                        old_metadata.get("confidence", 0.5)
                    ),
                    old_metadata.get("confidence", 0.5)
                )
                reason = update.get(
                    "reason",
                    old_metadata.get(
                        "reason",
                        "Updated based on the user's latest message."
                    )
                )

                # Never allow Gemini to modify a memory that was not
                # provided as a candidate by our application.
                if memory_id not in valid_ids:
                    continue

                if not memory_text:
                    continue

                try:
                    pending_result = (
                        self._reconcile_matching_memory_operation(
                            memory_text,
                            operation_types={"update"},
                        )
                    )
                    if pending_result is not None:
                        self._last_update_handled = True
                        update_status = {
                            "synchronized": "completed",
                            "partial": "partial",
                            "untracked": "partial",
                            "pending": "uncertain",
                        }[pending_result]
                        self._record_memory_activity(
                            "update",
                            update_status,
                            1,
                        )
                        if pending_result in {
                            "synchronized",
                            "partial",
                            "untracked",
                        }:
                            updated_any = True
                        continue

                    print(f"[MEMORY] Updating memory: {memory_id}")

                    messages = [
                        {
                            "role": "user",
                            "content": memory_text
                        }
                    ]

                    raw_evolution = old_metadata.get("evolution", {})
                    old_evolution = self._parse_evolution(raw_evolution)
                    history = []

                    if isinstance(old_evolution, dict):
                        evolution_metadata = dict(old_evolution)
                        existing_history = evolution_metadata.get("history", [])
                        if isinstance(existing_history, list):
                            history.extend(existing_history)
                        elif "history" in evolution_metadata:
                            evolution_metadata["legacy_history"] = (
                                evolution_metadata["history"]
                            )

                        if evolution_metadata.get("previous_memory"):
                            history.insert(
                                0,
                                {
                                    "memory_id": evolution_metadata.get(
                                        "previous_memory_id"
                                    ),
                                    "from": evolution_metadata.get(
                                        "previous_memory"
                                    ),
                                    "to": old_memory_text
                                }
                            )

                    elif isinstance(old_evolution, list):
                        evolution_metadata = {
                            "legacy_entries": list(old_evolution)
                        }
                    elif isinstance(raw_evolution, str):
                        evolution_metadata = {"legacy_value": raw_evolution}
                    else:
                        evolution_metadata = {}

                    history.append(
                        {
                            "memory_id": memory_id,
                            "from": old_memory_text,
                            "to": memory_text
                        }
                    )

                    write_result = self._write_memory_and_sync(
                        messages,
                        memory_text,
                        {
                            **old_metadata,
                            "importance": importance,
                            "confidence": confidence,
                            "reason": reason,
                            "source": old_metadata.get("source", "conversation"),
                            "evolution": json.dumps({
                                **evolution_metadata,
                                "history": history
                            }),
                        },
                        operation_type="update",
                        related_memory_ids=[memory_id],
                        create_relationships=False,
                    )

                    new_memory_id = write_result.get("memory_id")
                    self._last_update_handled = True
                    if write_result["status"] != "saved_graph_synced":
                        updated_any = write_result["status"] in {
                            "saved_graph_pending",
                            "saved_relationships_pending",
                        }
                        self._record_memory_activity(
                            "update",
                            (
                                "uncertain"
                                if write_result["status"] in {
                                    "outcome_uncertain",
                                    "timeout",
                                    "succeeded_without_memory_id",
                                }
                                else "partial"
                                if write_result["status"] in {
                                    "saved_graph_pending",
                                    "saved_relationships_pending",
                                }
                                else "failed"
                            ),
                            1,
                        )
                        print(
                            "[MEMORY] Memory update partially completed; "
                            "replacement was not fully "
                            "synchronized; the original memory was "
                            "preserved and reconciliation is required."
                        )
                        continue

                    deletion_confirmed = False
                    try:
                        delete_result = self.memory.delete(memory_id)
                        deletion_confirmed = self._delete_succeeded(delete_result)
                        if not deletion_confirmed:
                            print(
                                f"[MEMORY] Replacement {new_memory_id} was created, "
                                f"but deletion of {memory_id} was not confirmed."
                            )
                    except Exception as e:
                        print(
                            f"[MEMORY] Replacement {new_memory_id} was created, "
                            f"but deletion of {memory_id} failed: {e}"
                        )

                    graph_errors = []
                    recovery_warnings = []
                    graph_reconciled = False
                    graph_started = time.perf_counter()
                    if deletion_confirmed:
                        try:
                            reconciliation = self.graph.reconcile_memory_replacement(
                                self.user_id,
                                memory_id,
                                new_memory_id,
                                memory_text,
                            )
                            graph_reconciled = True
                            blocked_sources = reconciliation.get(
                                "blocked_sources",
                                [],
                            )
                            if blocked_sources:
                                recovery_warnings.append(
                                    "Pending recovery records were blocked "
                                    "because their expected target was replaced: "
                                    f"{blocked_sources}"
                                )
                        except Exception as e:
                            graph_errors.append(str(e))
                    else:
                        try:
                            self.graph.add_node(
                                self.user_id,
                                new_memory_id,
                                memory_text
                            )
                            graph_reconciled = True
                        except Exception as e:
                            graph_errors.append(str(e))
                    self._record_response_timing(
                        "graph_persistence",
                        graph_started,
                    )

                    if graph_reconciled:
                        try:
                            if not self.create_memory_relationships(
                                new_memory_id,
                                memory_text
                            ):
                                graph_errors.append("Relationship creation failed.")
                        except Exception as e:
                            graph_errors.append(str(e))

                    if (
                        deletion_confirmed
                        and not graph_errors
                        and not recovery_warnings
                    ):
                        self._record_memory_activity(
                            "update",
                            "completed",
                            1,
                        )
                        print("[MEMORY] Memory updated successfully.")
                    else:
                        self._record_memory_activity(
                            "update",
                            "partial",
                            1,
                        )
                        print(
                            f"[MEMORY] Memory update partially completed. "
                            f"Replacement ID: {new_memory_id}; "
                            f"old memory deletion confirmed: {deletion_confirmed}; "
                            f"graph reconciliation errors: {graph_errors}; "
                            f"recovery warnings: {recovery_warnings}"
                        )

                    # A confirmed replacement must not be stored a second time,
                    # even when deletion or graph reconciliation is partial.
                    updated_any = True

                except Exception as e:
                    self._record_memory_activity("update", "failed")
                    print(
                        f"\n[ERROR] Could not update memory {memory_id}: {e}\n"
                    )

            return updated_any

        except Exception as e:
            print(
                f"\n[ERROR] Memory update analysis failed: {e}\n"
            )

            if "response" in locals():


             return False
        finally:
            self._record_response_timing("memory_update", started_at)

    def decide_memory(self, message):
        """
        Decide whether the user's message contains information
        that should be stored as long-term memory.
        """

        prompt = f"""
You are the long-term memory decision system for an AI agent.

Your job is to decide whether the user's message contains information
that should be stored in long-term memory for future conversations.

Do NOT use a fixed list of categories or keywords to make this decision.
The user may provide any kind of information, and potentially important
information can appear in completely unexpected forms.

Instead, judge the information based on its long-term value and
potential usefulness in future conversations.

Store information when it is likely to remain useful beyond the current
conversation or situation. This includes information that helps the
agent understand the user, their history, their projects, their work,
their interests, their relationships, their decisions, their preferences,
their plans, their experiences, their knowledge, or important context
they have shared.

Also remember substantial context from conversations when that context
could help the agent understand or continue the user's work in a future
conversation. If the user explains a project, system, idea, workflow,
research, plan, or other ongoing subject in meaningful detail, preserve
the important information from it rather than remembering only isolated
sentences.

The information does NOT need to be permanent to be useful. Information
can still be worth remembering if it is likely to remain relevant for a
reasonable period of time or could help the agent provide better context
in a future conversation.

Distinguish between a momentary state and information that has future
relevance.

Do NOT store information merely because it is happening right now when
it has no meaningful relevance beyond the current moment.

For example:

"Sandeep is eating right now."
→ temporary state → normally do not remember.

"Sandeep is sleeping right now."
→ temporary state → normally do not remember.

"It is raining outside right now."
→ temporary observation → normally do not remember.

"Someone is watching TV right now."
→ temporary state → normally do not remember.

However, remember information that describes a future event, plan,
deadline, intention, expectation, or upcoming activity, even if that
information is time-sensitive.

For example:

"ISRO is going to do a launch next month."
→ future event → remember.

"My exam is next month."
→ future deadline → remember.

"I am planning to visit Delhi next month."
→ future plan → remember.

"My project presentation is on Friday."
→ upcoming event → remember.

The fact that information will eventually become outdated does NOT
automatically make it unworthy of memory. Consider whether it could be
useful in a future conversation before deciding.

The same principle applies to information that is not explicitly about
the user. Important information about projects, organizations, people,
events, research, or other subjects can be worth remembering when it
provides meaningful context for future conversations.

For example:

"ISRO is working on a new mission."
→ potentially useful subject context → consider remembering.

"The launch is scheduled for next month."
→ future event → remember.

"ISRO launched a rocket today."
→ current event → normally do not remember unless the context makes
it particularly important or useful later.

The important distinction is:

Momentary and contextless → normally do not remember.
Meaningful or future-relevant → remember.
Potentially useful context → consider remembering.

When a message contains a mixture of temporary and meaningful
information, remember the meaningful information and ignore the
temporary details.

Prefer remembering useful information over aggressively filtering it.
Do not be overly restrictive.

When uncertain, ask yourself:

"If the user talks to this agent again days, weeks, or months from now,
could knowing this information make the agent substantially more useful?"

If yes, remember it.

If the information is only useful for the immediate moment and is
unlikely to matter later, do not remember it.

Do not invent information or infer facts that the user did not actually
provide.

Return ONLY valid JSON in this exact format:

{{
    "should_remember": true
}}

or

{{
    "should_remember": false
}}

User message:

"{message}"
"""

        started_at = time.perf_counter()
        try:
            with timed_operation("Gemini memory decision"):
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    )
                )

            response_text = response.text.strip()

            if response_text.startswith("```"):
             response_text = response_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(response_text)

            decision = result.get("should_remember")
            if isinstance(decision, bool):
                self._record_memory_activity(
                    "decision",
                    "completed" if decision else "none",
                )
            else:
                self._record_memory_activity("decision", "failed")
                return False
            return decision

        except Exception as e:
            self._record_memory_activity("decision", "failed")
            print(
                f"\n[ERROR] Gemini memory decision failed: {e}\n"
            )
            return False
        finally:
            self._record_response_timing("memory_decision", started_at)

    def respond(self, message):
        self.last_response_metadata = {
            "memory_activity": [],
            "timings_ms": {},
        }
        started_at = time.perf_counter()
        try:
            with timed_operation("User-message handling"):
                return self._respond(message)
        finally:
            self._record_response_timing("agent_execution", started_at)

    def _respond(self, message):
        """
        Generate a response using short-term
        and long-term memory.
        """

        # Get relevant long-term memories.
        memories = self.recall(message)

        # Check every message for possible memory updates.
        # Mem0 finds likely related memories first, then Gemini decides
        # whether the new message actually changes any of them.
        candidate_memories = []

        if memories and "results" in memories:
            candidate_memories = [
                item
                for item in memories["results"]
                if item.get("score", 0) >= 0.25
            ]

            candidate_memories = sorted(
                candidate_memories,
                key=lambda item: item.get("score", 0),
                reverse=True
            )[:10]

        # If semantic search does not return a strong enough candidate,
        # fall back to the user's stored memories so Gemini can resolve
        # short contextual updates such as "move it to 12".
        if (
            not candidate_memories
            and self._needs_memory_fallback(message)
        ):
            try:
                memory_list_started_at = time.perf_counter()
                all_memories = self.memory.get_all(self.user_id)
                if all_memories and "results" in all_memories:
                    candidate_memories = all_memories["results"][:10]
            except Exception as e:
                self._record_memory_activity("update", "failed")
                print(f"[ERROR] Could not load memories for update analysis: {e}")
            finally:
                self._record_response_timing(
                    "memory_fallback_list",
                    memory_list_started_at,
                )

        # Historical memories should not be updated again.
        # If a candidate is referenced by another candidate's
        # evolution metadata, it is an older version in that timeline.
        historical_ids = set()

        for item in candidate_memories:
            historical_ids.update(
                self._historical_memory_ids(
                    item.get("metadata", {}).get("evolution", {})
                )
            )

        candidate_memories = [
            item
            for item in candidate_memories
            if item.get("id") not in historical_ids
        ]

        existing_update_events = sum(
            event["action"] == "update"
            for event in self.last_response_metadata["memory_activity"]
        )
        was_updated = self.update_memory(message, candidate_memories)
        update_handled = was_updated or getattr(
            self,
            "_last_update_handled",
            False,
        )
        updated_event_count = sum(
            event["action"] == "update"
            for event in self.last_response_metadata["memory_activity"]
        )
        if updated_event_count == existing_update_events:
            self._record_memory_activity(
                "update",
                "completed" if was_updated else "none",
                1 if was_updated else None,
            )

        # If a memory changed, retrieve again so the response uses the
        # newly stored version instead of the outdated one.
        if was_updated:
            memories = self.recall(message)

        memory_context = ""

        if memories and "results" in memories:

            # Historical versions belong to the evolution timeline and
            # should not be treated as separate current memories.
            historical_ids = set()

            for item in memories["results"]:
                historical_ids.update(
                    self._historical_memory_ids(
                        item.get("metadata", {}).get("evolution")
                    )
                )

            relevant_memories = [
                item
                for item in memories["results"]
                if (
                    item.get("score", 0) >= 0.15
                    and item.get("id") not in historical_ids
                )
            ]

            # Combine semantic relevance, importance, and recency.
            for item in relevant_memories:

                # Semantic relevance from Mem0.
                relevance_score = float(
                    item.get("score", 0)
                )

                # Importance assigned when the memory was created.
                importance_score = float(
                    item.get("metadata", {}).get(
                        "importance",
                        5
                    )
                )

                # Convert importance from 1-10 to 0-1.
                importance_score = importance_score / 10

                # Calculate how recent the memory is.
                created_at = item.get("created_at")

                recency_score = 0.5

                if created_at:
                    try:
                        from datetime import datetime, timezone

                        created_time = datetime.fromisoformat(
                            created_at.replace(
                                "Z",
                                "+00:00"
                            )
                        )

                        now = datetime.now(timezone.utc)

                        age_days = (
                            now - created_time
                        ).total_seconds() / 86400

                        # Memory loses half its recency score
                        # every 30 days.
                        recency_score = 2 ** (
                            -age_days / 30
                        )

                    except Exception:
                        recency_score = 0.5

                # Final memory ranking.
                item["final_score"] = (
                    relevance_score * 0.60
                    + importance_score * 0.25
                    + recency_score * 0.15
                )

            # Keep only the top 5 memories by final score.
            relevant_memories = sorted(
                relevant_memories,
                key=lambda item: item.get(
                    "final_score",
                    0
                ),
                reverse=True
            )[:5]

            memory_context = "\n".join(
                item["memory"]
                for item in relevant_memories
            )

        # Add user's message to short-term memory.
        self.conversation_history.append({
            "role": "user",
            "text": message
        })

        # Keep only recent conversation.
        recent_history = self.conversation_history[
            -self.max_history:
        ]

        conversation_context = "\n".join(
            f"{item['role']}: {item['text']}"
            for item in recent_history
        )

        # Give Gemini both types of context.
        system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Relevant long-term memories:
{memory_context}

Recent conversation:
{conversation_context}

Current user message:
{message}

Use both types of context when relevant.

The current user message is the most important input.
Use recent conversation to understand context and references.

When recent conversation contains newer information that conflicts
with older information, always prefer the most recent information.

When answering a question about the current state of something,
use the most recent information as the current truth.

Do not bring up older states unless the user asks about them.
Do not add unnecessary commentary about changes from earlier.

Use long-term memories silently.
Never mention memories or how you know something.
Never say "based on what you told me", "you told me",
"according to my memory", or similar phrases.

Do not invent information.
Answer naturally.
"""

        response_started_at = time.perf_counter()
        try:
            with timed_operation("Gemini response generation"):
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=system_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    )
                )
        finally:
            self._record_response_timing(
                "gemini_response",
                response_started_at,
            )


        answer = response.text

        # Add AI response to short-term memory.
        self.conversation_history.append({
            "role": "assistant",
            "text": answer
        })

        # Store normal long-term information only when the message did not
        # update an existing memory. Updated memories have already been
        # replaced by update_memory().
        should_remember = (
            False if update_handled else self.decide_memory(message)
        )
        print(f"[DEBUG] Should remember: {should_remember}")


        if not update_handled and should_remember:
            self.remember(message, answer)
        elif not update_handled and not any(
            event["action"] == "save"
            for event in self.last_response_metadata["memory_activity"]
        ):
            self._record_memory_activity("save", "none")

        return answer