import copy
import contextlib
import errno
import hashlib
import json
import os
import tempfile
import threading
import time

from app.performance import timed_operation


class KnowledgeGraph:
    RELATIONSHIP_RECOVERY_KEY = "relationship_recovery"
    MAX_RECOVERY_RECORDS = 200
    MAX_RECOVERY_ATTEMPTS = 3
    MAX_EXPECTED_EDGES = 10
    MAX_RECOVERY_ERROR_LENGTH = 240
    FILE_LOCK_TIMEOUT_SECONDS = 5
    FILE_LOCK_POLL_SECONDS = 0.05
    RECOVERY_STATUSES = {
        "pending_generation",
        "pending_persistence",
        "succeeded",
        "succeeded_empty",
        "superseded",
        "failed",
        "blocked_missing_endpoint",
    }

    class StateConflictError(RuntimeError):
        """Raised when the graph file changed since this instance loaded it."""

    class FileLockError(RuntimeError):
        """Raised when the graph file lock cannot be acquired or released."""

    def __init__(self, file_path="graph.json"):
        self.path_file = os.path.realpath(os.path.abspath(file_path))
        self._lock = threading.RLock()
        self._file_version = None
        self.graph = self._load()

    def _load(self):
        if not os.path.exists(self.path_file):
            return {}

        try:
            with open(self.path_file, "rb") as file:
                contents = file.read()
            self._file_version = hashlib.sha256(contents).digest()
            return json.loads(contents.decode("utf-8"))

        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Error loading graph from {self.path_file}: {e}"
            )
            return {}

    def _current_file_version(self):
        try:
            with open(self.path_file, "rb") as file:
                return hashlib.sha256(file.read()).digest()
        except FileNotFoundError:
            return None

    def _ensure_file_version_matches(self):
        if self._current_file_version() != self._file_version:
            raise self.StateConflictError(
                "Graph file changed since it was loaded; reload before writing."
            )

    @classmethod
    @contextlib.contextmanager
    def _file_lock(cls, graph_path):
        lock_path = os.path.realpath(os.path.abspath(graph_path)) + ".lock"
        try:
            lock_file = open(lock_path, "a+b")
        except OSError as error:
            raise cls.FileLockError(
                "Could not open the graph file lock."
            ) from error
        locked = False
        try:
            try:
                if os.fstat(lock_file.fileno()).st_size == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
            except OSError as error:
                raise cls.FileLockError(
                    "Could not initialize the graph file lock."
                ) from error
            deadline = time.monotonic() + cls.FILE_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    lock_file.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(
                            lock_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    locked = True
                    break
                except OSError as error:
                    contention_errors = {
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EDEADLK,
                    }
                    if (
                        error.errno not in contention_errors
                        and getattr(error, "winerror", None) not in {33, 36}
                    ):
                        raise cls.FileLockError(
                            "Could not acquire the graph file lock."
                        ) from error
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise cls.FileLockError(
                            "Timed out waiting for the graph file lock."
                        ) from error
                    time.sleep(min(cls.FILE_LOCK_POLL_SECONDS, remaining))
            yield
        finally:
            if locked:
                try:
                    lock_file.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(
                            lock_file.fileno(),
                            msvcrt.LK_UNLCK,
                            1,
                        )
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError as error:
                    lock_file.close()
                    raise cls.FileLockError(
                        "Could not release the graph file lock."
                    ) from error
            lock_file.close()

    def save(self, graph=None):
        with timed_operation("Graph persistence"):
            return self._save(graph)

    def _save(self, graph=None):
        with self._lock:
            with self._file_lock(self.path_file):
                self._ensure_file_version_matches()
                graph_to_save = copy.deepcopy(
                    self.graph if graph is None else graph
                )
                contents = json.dumps(graph_to_save, indent=2).encode("utf-8")
                directory = os.path.dirname(self.path_file)
                descriptor, temporary_path = tempfile.mkstemp(
                    prefix=".graph-",
                    suffix=".tmp",
                    dir=directory
                )
                try:
                    with os.fdopen(descriptor, "wb") as file:
                        file.write(contents)
                    self._ensure_file_version_matches()
                    os.replace(temporary_path, self.path_file)
                except Exception:
                    try:
                        os.unlink(temporary_path)
                    except FileNotFoundError:
                        pass
                    raise
                self.graph = graph_to_save
                self._file_version = hashlib.sha256(contents).digest()

    def _commit_user_graph(self, user_id, user_graph):
        updated_graph = dict(self.graph)
        updated_graph[user_id] = user_graph
        self.save(updated_graph)
        self.graph = updated_graph

    @staticmethod
    def _valid_recovery_record(record):
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("source_id"), str)
            or not record["source_id"].strip()
            or len(record["source_id"]) > 256
            or record.get("status") not in KnowledgeGraph.RECOVERY_STATUSES
            or type(record.get("attempts")) is not int
            or not 0 <= record["attempts"] <= KnowledgeGraph.MAX_RECOVERY_ATTEMPTS
            or not isinstance(record.get("error"), (str, type(None)))
            or (
                isinstance(record.get("error"), str)
                and len(record["error"]) > KnowledgeGraph.MAX_RECOVERY_ERROR_LENGTH
            )
            or not isinstance(record.get("expected_edges"), list)
            or len(record["expected_edges"]) > KnowledgeGraph.MAX_EXPECTED_EDGES
        ):
            return False

        for edge in record["expected_edges"]:
            if (
                not isinstance(edge, dict)
                or not isinstance(edge.get("target_id"), str)
                or not edge["target_id"].strip()
                or len(edge["target_id"]) > 256
                or not isinstance(edge.get("relationship"), str)
                or not edge["relationship"].strip()
                or len(edge["relationship"]) > 256
            ):
                return False
        if (
            record["status"] == "pending_persistence"
            and not record["expected_edges"]
        ):
            return False
        if (
            record["status"] in {
                "pending_generation",
                "failed",
                "succeeded",
                "succeeded_empty",
            }
            and record["expected_edges"]
        ):
            return False
        if record["status"] == "superseded" and (
            not isinstance(record.get("replacement_id"), str)
            or not record["replacement_id"].strip()
            or len(record["replacement_id"]) > 256
        ):
            return False
        return True

    def restore_local_recovery_source(
        self,
        user_id,
        replacement_id,
        memory_text,
    ):
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(replacement_id, str)
            or not replacement_id.strip()
            or len(replacement_id) > 256
            or not isinstance(memory_text, str)
            or not memory_text.strip()
        ):
            return {"status": "invalid_input"}

        with self._lock:
            try:
                user_graph = self._user_graph_copy(user_id)
            except ValueError:
                return {"status": "invalid_user_graph"}

            recovery = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY, {})
            if not isinstance(recovery, dict):
                return {"status": "invalid_queue"}
            if len(recovery) > self.MAX_RECOVERY_RECORDS:
                return {"status": "invalid_queue"}
            if any(
                not self._valid_recovery_record(record)
                or key != record.get("source_id")
                for key, record in recovery.items()
            ):
                return {"status": "invalid_queue"}

            provenance_records = [
                record
                for key, record in recovery.items()
                if (
                    key == record.get("source_id")
                    and record["status"] == "superseded"
                    and record.get("replacement_id") == replacement_id
                )
            ]
            if len(provenance_records) != 1:
                return {"status": "provenance_not_found"}

            node_exists = replacement_id in self._node_ids(user_graph)
            recovery_exists = replacement_id in recovery
            if node_exists or recovery_exists:
                return {
                    "status": "already_present",
                    "node_exists": node_exists,
                    "recovery_record_exists": recovery_exists,
                }

            if len(recovery) >= self.MAX_RECOVERY_RECORDS:
                return {"status": "queue_full"}

            user_graph["nodes"].append(
                {
                    "id": replacement_id,
                    "memory_text": memory_text,
                }
            )
            recovery[replacement_id] = {
                "source_id": replacement_id,
                "status": "pending_generation",
                "attempts": 0,
                "error": None,
                "expected_edges": [],
            }
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = recovery

            updated_graph = dict(self.graph)
            updated_graph[user_id] = user_graph
            self.save(updated_graph)
            self.graph = updated_graph
            return {"status": "restored"}

    def _user_graph_copy(self, user_id):
        current = self.graph.get(
            user_id,
            {
                "nodes": [],
                "relationships": []
            }
        )
        if (
            not isinstance(current, dict)
            or not isinstance(current.get("nodes"), list)
            or not isinstance(current.get("relationships"), list)
        ):
            raise ValueError("Cannot update an invalid user graph.")
        return copy.deepcopy(current)

    @staticmethod
    def _node_ids(user_graph):
        return {
            node.get("id")
            for node in user_graph["nodes"]
            if (
                isinstance(node, dict)
                and isinstance(node.get("id"), str)
                and node["id"].strip()
            )
        }

    def get_relationship_recovery(self, user_id, user_graph=None):
        with self._lock:
            if user_graph is None:
                user_graph = self.graph.get(user_id, {})
            return self.validate_relationship_recovery(user_graph)

    @classmethod
    def validate_relationship_recovery(cls, user_graph):
        if not isinstance(user_graph, dict):
            return {
                "valid": False,
                "records": [],
                "invalid_records": [str(user_graph)],
                "blocked_count": 1,
                "queue_size": 0,
                "error": "User graph is malformed.",
            }

        queue = user_graph.get(cls.RELATIONSHIP_RECOVERY_KEY, {})
        if not isinstance(queue, dict):
            return {
                "valid": False,
                "records": [],
                "invalid_records": [
                    str(queue)[:cls.MAX_RECOVERY_ERROR_LENGTH]
                ],
                "blocked_count": 1,
                "queue_size": 0,
                "error": "Relationship recovery queue is malformed.",
            }
        if len(queue) > cls.MAX_RECOVERY_RECORDS:
            queue_size = len(queue)
            return {
                "valid": False,
                "records": [],
                "invalid_records": [],
                "blocked_count": queue_size,
                "queue_size": queue_size,
                "error": (
                    f"Relationship recovery queue has {queue_size} records, "
                    f"exceeding the limit of {cls.MAX_RECOVERY_RECORDS}. "
                    "No records were changed or processed. Review and resolve "
                    "records manually before retrying recovery."
                ),
            }

        records = []
        invalid_records = []
        for source_id, record in queue.items():
            if not cls._valid_recovery_record(record) or source_id != record.get(
                "source_id"
            ):
                invalid_records.append(str(source_id))
                continue
            records.append(copy.deepcopy(record))

        return {
            "valid": True,
            "records": records,
            "invalid_records": invalid_records,
            "blocked_count": len(invalid_records),
            "queue_size": len(queue),
            "error": None,
        }

    def create_relationship_recovery(self, user_id, source_id):
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(source_id, str)
            or not source_id.strip()
            or len(source_id) > 256
        ):
            return False

        with self._lock:
            user_graph = self._user_graph_copy(user_id)
            if source_id not in self._node_ids(user_graph):
                return False

            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY, {})
            if not isinstance(queue, dict):
                return False
            if source_id in queue:
                return self._valid_recovery_record(queue[source_id])
            if len(queue) >= self.MAX_RECOVERY_RECORDS:
                return False

            queue[source_id] = {
                "source_id": source_id,
                "status": "pending_generation",
                "attempts": 0,
                "error": None,
                "expected_edges": [],
            }
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return True

    def begin_relationship_attempt(
        self,
        user_id,
        source_id,
        max_attempts=MAX_RECOVERY_ATTEMPTS,
        expected_user_graph=None,
    ):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer.")
        max_attempts = min(max_attempts, self.MAX_RECOVERY_ATTEMPTS)
        with self._lock:
            if (
                expected_user_graph is not None
                and self.graph.get(user_id, {}) != expected_user_graph
            ):
                raise self.StateConflictError(
                    "Recovery state changed after eligibility was evaluated."
                )
            user_graph = self._user_graph_copy(user_id)
            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            record = queue.get(source_id) if isinstance(queue, dict) else None
            if not self._valid_recovery_record(record):
                return {"status": "invalid", "record": None}
            if source_id not in self._node_ids(user_graph):
                return {"status": "blocked_missing_endpoint", "record": record}
            if record["status"] not in {"pending_generation", "failed"}:
                return {"status": "not_eligible", "record": copy.deepcopy(record)}
            if record["attempts"] >= max_attempts:
                return {"status": "capped", "record": copy.deepcopy(record)}

            record["attempts"] += 1
            record["status"] = "pending_generation"
            record["error"] = None
            queue[source_id] = record
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return {"status": "ready", "record": copy.deepcopy(record)}

    def set_relationship_recovery_status(
        self,
        user_id,
        source_id,
        status,
        error=None
    ):
        if status not in self.RECOVERY_STATUSES:
            raise ValueError("Unsupported relationship recovery status.")
        if status == "superseded":
            raise ValueError(
                "Superseded status requires a confirmed replacement ID."
            )

        with self._lock:
            user_graph = self._user_graph_copy(user_id)
            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            record = queue.get(source_id) if isinstance(queue, dict) else None
            if not self._valid_recovery_record(record):
                return False

            if (
                status == "pending_persistence"
                and not record["expected_edges"]
            ):
                return False
            record["status"] = status
            record["error"] = (
                str(error)[:self.MAX_RECOVERY_ERROR_LENGTH]
                if error
                else None
            )
            if status in {"succeeded", "succeeded_empty"}:
                record["expected_edges"] = []
            queue[source_id] = record
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return True

    def set_pending_relationship_edges(self, user_id, source_id, edges):
        if (
            not isinstance(edges, list)
            or not edges
            or len(edges) > self.MAX_EXPECTED_EDGES
            or any(
                not isinstance(edge, dict)
                or not isinstance(edge.get("target_id"), str)
                or not edge["target_id"].strip()
                or len(edge["target_id"]) > 256
                or not isinstance(edge.get("relationship"), str)
                or not edge["relationship"].strip()
                or len(edge["relationship"]) > 256
                for edge in edges
            )
        ):
            return False

        with self._lock:
            user_graph = self._user_graph_copy(user_id)
            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            record = queue.get(source_id) if isinstance(queue, dict) else None
            if (
                not self._valid_recovery_record(record)
                or record["status"] not in {"pending_generation", "failed"}
            ):
                return False

            node_ids = self._node_ids(user_graph)
            if source_id not in node_ids or any(
                edge["target_id"] not in node_ids for edge in edges
            ):
                return False

            record["status"] = "pending_persistence"
            record["error"] = None
            record["expected_edges"] = [
                {
                    "target_id": edge["target_id"],
                    "relationship": edge["relationship"],
                }
                for edge in edges
            ]
            queue[source_id] = record
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return True

    def complete_relationship_recovery(
        self,
        user_id,
        source_id,
        expected_user_graph=None,
    ):
        with self._lock:
            if (
                expected_user_graph is not None
                and self.graph.get(user_id, {}) != expected_user_graph
            ):
                raise self.StateConflictError(
                    "Recovery state changed after eligibility was evaluated."
                )
            user_graph = self._user_graph_copy(user_id)
            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            record = queue.get(source_id) if isinstance(queue, dict) else None
            if (
                not self._valid_recovery_record(record)
                or record["status"] != "pending_persistence"
                or not isinstance(record.get("expected_edges"), list)
                or len(record["expected_edges"]) > self.MAX_EXPECTED_EDGES
            ):
                return {"status": "invalid"}

            node_ids = self._node_ids(user_graph)
            if source_id not in node_ids or any(
                not isinstance(edge, dict)
                or not isinstance(edge.get("target_id"), str)
                or not edge["target_id"].strip()
                or edge.get("target_id") not in node_ids
                or not isinstance(edge.get("relationship"), str)
                or not edge["relationship"].strip()
                for edge in record["expected_edges"]
            ):
                return {"status": "blocked_missing_endpoint"}

            relationships = user_graph["relationships"]
            for edge in record["expected_edges"]:
                candidate = {
                    "source": source_id,
                    "target": edge["target_id"],
                    "relationship": edge["relationship"],
                }
                if candidate not in relationships:
                    relationships.append(candidate)

            record["status"] = "succeeded"
            record["error"] = None
            record["expected_edges"] = []
            queue[source_id] = record
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return {"status": "succeeded"}

    def finish_empty_relationship_recovery(self, user_id, source_id):
        with self._lock:
            user_graph = self._user_graph_copy(user_id)
            queue = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            record = queue.get(source_id) if isinstance(queue, dict) else None
            if (
                not self._valid_recovery_record(record)
                or record["status"] not in {"pending_generation", "failed"}
            ):
                return False
            record["status"] = "succeeded_empty"
            record["error"] = None
            record["expected_edges"] = []
            queue[source_id] = record
            user_graph[self.RELATIONSHIP_RECOVERY_KEY] = queue
            self._commit_user_graph(user_id, user_graph)
            return True

    def get_user_graph(self, user_id):
        return self.graph.setdefault(
            user_id,
            {
                "nodes": [],
                "relationships": []
            }
        )

    def add_node(
        self,
        user_id,
        memory_id,
        memory_text
    ):
        with self._lock:
            user_graph = copy.deepcopy(
                self.graph.get(
                    user_id,
                    {
                        "nodes": [],
                        "relationships": []
                    }
                )
            )

            for node in user_graph["nodes"]:
                if node["id"] == memory_id:
                    node["memory_text"] = memory_text
                    self._commit_user_graph(user_id, user_graph)
                    return

            user_graph["nodes"].append(
                {
                    "id": memory_id,
                    "memory_text": memory_text
                }
            )

            self._commit_user_graph(user_id, user_graph)

    def add_relationship(
        self,
        user_id,
        source_id,
        target_id,
        relationship
    ):
        """Add an edge if both endpoints exist; return False when rejected."""

        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(target_id, str)
            or not target_id.strip()
            or not isinstance(relationship, str)
            or not relationship.strip()
        ):
            return False

        with self._lock:
            user_graph = self.graph.get(user_id)
            if (
                not isinstance(user_graph, dict)
                or not isinstance(user_graph.get("nodes"), list)
                or not isinstance(user_graph.get("relationships"), list)
            ):
                return False

            node_ids = {
                node.get("id")
                for node in user_graph["nodes"]
                if (
                    isinstance(node, dict)
                    and isinstance(node.get("id"), str)
                    and node["id"].strip()
                )
            }
            if source_id not in node_ids or target_id not in node_ids:
                return False

            for edge in user_graph["relationships"]:
                if (
                    isinstance(edge, dict)
                    and edge.get("source") == source_id
                    and edge.get("target") == target_id
                    and edge.get("relationship") == relationship
                ):
                    return True

            updated_user_graph = copy.deepcopy(user_graph)
            updated_user_graph["relationships"].append(
                {
                    "source": source_id,
                    "target": target_id,
                    "relationship": relationship
                }
            )

            self._commit_user_graph(user_id, updated_user_graph)
            return True

    def delete_node(
        self,
        user_id,
        memory_id
    ):
        with self._lock:
            user_graph = copy.deepcopy(
                self.graph.get(
                    user_id,
                    {
                        "nodes": [],
                        "relationships": []
                    }
                )
            )

            user_graph["nodes"] = [
                node
                for node in user_graph["nodes"]
                if node["id"] != memory_id
            ]

            user_graph["relationships"] = [
                relationship
                for relationship in user_graph["relationships"]
                if (
                    relationship["source"] != memory_id
                    and relationship["target"] != memory_id
                )
            ]

            self._commit_user_graph(user_id, user_graph)

    def reconcile_memory_replacement(
        self,
        user_id,
        old_memory_id,
        replacement_id,
        replacement_text,
    ):
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(old_memory_id, str)
            or not old_memory_id.strip()
            or not isinstance(replacement_id, str)
            or not replacement_id.strip()
            or not isinstance(replacement_text, str)
            or not replacement_text.strip()
        ):
            raise ValueError("Memory replacement details are invalid.")

        reason = (
            f"Source memory was replaced by {replacement_id}; "
            "relationships were not redirected."
        )[:self.MAX_RECOVERY_ERROR_LENGTH]

        with self._lock:
            user_graph = self._user_graph_copy(user_id)
            if old_memory_id != replacement_id:
                user_graph["nodes"] = [
                    node
                    for node in user_graph["nodes"]
                    if not (
                        isinstance(node, dict)
                        and node.get("id") == old_memory_id
                    )
                ]
                user_graph["relationships"] = [
                    edge
                    for edge in user_graph["relationships"]
                    if not (
                        isinstance(edge, dict)
                        and (
                            edge.get("source") == old_memory_id
                            or edge.get("target") == old_memory_id
                        )
                    )
                ]

            replacement_node = next(
                (
                    node for node in user_graph["nodes"]
                    if isinstance(node, dict)
                    and node.get("id") == replacement_id
                ),
                None,
            )
            if replacement_node is None:
                user_graph["nodes"].append(
                    {
                        "id": replacement_id,
                        "memory_text": replacement_text,
                    }
                )
            else:
                replacement_node["memory_text"] = replacement_text

            blocked_sources = []
            superseded = False
            recovery = user_graph.get(self.RELATIONSHIP_RECOVERY_KEY)
            if isinstance(recovery, dict) and old_memory_id != replacement_id:
                old_record = recovery.get(old_memory_id)
                if (
                    self._valid_recovery_record(old_record)
                    and old_record.get("source_id") == old_memory_id
                ):
                    old_record["status"] = "superseded"
                    old_record["replacement_id"] = replacement_id
                    old_record["error"] = reason
                    superseded = True

                for source_id, record in recovery.items():
                    if (
                        source_id == old_memory_id
                        or not self._valid_recovery_record(record)
                        or record["status"] not in {
                            "pending_generation",
                            "pending_persistence",
                            "failed",
                        }
                    ):
                        continue
                    expected_edges = record.get("expected_edges", [])
                    if any(
                        edge.get("target_id") == old_memory_id
                        for edge in expected_edges
                    ):
                        record["status"] = "blocked_missing_endpoint"
                        record["error"] = reason
                        blocked_sources.append(source_id)

            self._commit_user_graph(user_id, user_graph)
            return {
                "superseded": superseded,
                "blocked_sources": blocked_sources,
            }

    def sync_user_graph(
        self,
        user_id,
        memories
    ):
        with timed_operation("Graph synchronization"):
            return self._sync_user_graph(user_id, memories)

    def _sync_user_graph(
        self,
        user_id,
        memories
    ):
        """
        Synchronize the user's knowledge graph with
        the currently existing Mem0 memories.
        """

        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(memories, dict)
            or not isinstance(memories.get("results"), list)
            or type(memories.get("count")) is not int
            or memories["count"] != len(memories["results"])
        ):
            raise ValueError(
                "Cannot synchronize graph from an incomplete or invalid memory listing."
            )

        current_memories = {}

        for memory in memories.get("results", []):
            if (
                not isinstance(memory, dict)
                or not isinstance(memory.get("id"), str)
                or not memory["id"].strip()
                or not isinstance(memory.get("memory"), str)
                or not memory["memory"].strip()
            ):
                raise ValueError(
                    "Cannot synchronize graph from a memory without a valid ID and usable text."
                )

            memory_id = memory["id"]
            if memory_id in current_memories:
                raise ValueError(
                    "Cannot synchronize graph from duplicate memory IDs."
                )
            current_memories[memory_id] = memory["memory"].strip()

        valid_ids = set(
            current_memories.keys()
        )

        with self._lock:
            user_graph = self.graph.get(user_id)
            if user_graph is None:
                user_graph = {
                    "nodes": [],
                    "relationships": []
                }
            elif (
                not isinstance(user_graph, dict)
                or not isinstance(user_graph.get("relationships"), list)
            ):
                raise ValueError(
                    "Cannot synchronize an invalid existing graph."
                )

            valid_relationships = [
                relationship
                for relationship in user_graph["relationships"]
                if (
                    isinstance(relationship, dict)
                    and isinstance(relationship.get("source"), str)
                    and relationship["source"] in valid_ids
                    and isinstance(relationship.get("target"), str)
                    and relationship["target"] in valid_ids
                )
            ]
            current_nodes = [
                {
                    "id": memory_id,
                    "memory_text": memory_text
                }
                for memory_id, memory_text
                in current_memories.items()
            ]

            updated_user_graph = copy.deepcopy(user_graph)
            updated_user_graph["nodes"] = current_nodes
            updated_user_graph["relationships"] = valid_relationships
            recovery = updated_user_graph.get(
                self.RELATIONSHIP_RECOVERY_KEY
            )
            if isinstance(recovery, dict):
                current_ids = set(current_memories)
                for source_id, record in recovery.items():
                    if not self._valid_recovery_record(record):
                        continue
                    if record["status"] in {
                        "succeeded",
                        "succeeded_empty",
                        "superseded",
                    }:
                        continue
                    expected_edges = record.get("expected_edges", [])
                    has_missing_target = (
                        isinstance(expected_edges, list)
                        and any(
                            isinstance(edge, dict)
                            and edge.get("target_id") not in current_ids
                            for edge in expected_edges
                        )
                    )
                    if (
                        source_id not in current_ids
                        or has_missing_target
                    ):
                        record["status"] = "blocked_missing_endpoint"
                        record["error"] = (
                            "A source or target memory is absent from the "
                            "current complete listing."
                        )[:self.MAX_RECOVERY_ERROR_LENGTH]

            self._commit_user_graph(user_id, updated_user_graph)