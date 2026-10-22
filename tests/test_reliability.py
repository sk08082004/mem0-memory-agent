import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def _install_service_import_stubs():
    mem0 = types.ModuleType("mem0")
    mem0.MemoryClient = object
    sys.modules["mem0"] = mem0

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = lambda **kwargs: None
    google.genai = genai

    genai_types = types.ModuleType("google.genai.types")
    for name in (
        "GenerateContentConfig",
        "ThinkingConfig",
        "AutomaticFunctionCallingConfig",
        "HttpOptions",
        "HttpRetryOptions",
    ):
        setattr(genai_types, name, lambda **kwargs: kwargs)
    genai.types = genai_types

    sys.modules["google"] = google
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = genai_types


_install_service_import_stubs()

from app.agent import Agent
from app.graph import KnowledgeGraph
from app.main import (
    create_agent,
    load_user,
    main,
    normalize_user_id,
    save_user,
)
from app.memory import MemoryManager
import app.main as main_module


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FakeMemory:
    def __init__(
        self,
        add_result=None,
        delete_result=None,
        add_error=None,
        delete_error=None,
    ):
        self.add_result = add_result
        self.delete_result = delete_result
        self.add_error = add_error
        self.delete_error = delete_error
        self.added = []
        self.deleted = []

    def add(self, messages, user_id, metadata=None):
        self.added.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
            }
        )
        if self.add_error:
            raise self.add_error
        return self.add_result

    def delete(self, memory_id):
        self.deleted.append(memory_id)
        if self.delete_error:
            raise self.delete_error
        return self.delete_result


class FakeGraph:
    MAX_EXPECTED_EDGES = 10
    MAX_RECOVERY_ERROR_LENGTH = 240

    def __init__(self, node_error=None, relationship_result=True):
        self.nodes = []
        self.edges = []
        self.deleted_nodes = []
        self.node_error = node_error
        self.relationship_result = relationship_result
        self.sync_calls = []
        self.recovery = {}

    def add_node(self, user_id, memory_id, memory_text):
        if self.node_error:
            raise self.node_error
        self.nodes.append(
            {
                "user_id": user_id,
                "id": memory_id,
                "memory_text": memory_text,
            }
        )

    def delete_node(self, user_id, memory_id):
        self.deleted_nodes.append((user_id, memory_id))

    def get_user_graph(self, user_id):
        return {
            "nodes": [
                {
                    "id": node["id"],
                    "memory_text": node["memory_text"],
                }
                for node in self.nodes
                if node["user_id"] == user_id
            ],
            "relationships": list(self.edges),
        }

    def add_relationship(self, user_id, source_id, target_id, relationship):
        if not self.relationship_result:
            return False
        self.edges.append(
            {
                "source": source_id,
                "target": target_id,
                "relationship": relationship,
            }
        )
        return True

    def sync_user_graph(self, user_id, memories):
        self.sync_calls.append((user_id, copy.deepcopy(memories)))

    def create_relationship_recovery(self, user_id, source_id):
        if not any(
            node["user_id"] == user_id and node["id"] == source_id
            for node in self.nodes
        ):
            return False
        self.recovery.setdefault(
            (user_id, source_id),
            {
                "source_id": source_id,
                "status": "pending_generation",
                "attempts": 0,
                "error": None,
                "expected_edges": [],
            },
        )
        return True

    def begin_relationship_attempt(self, user_id, source_id, max_attempts=3):
        record = self.recovery.get((user_id, source_id))
        if not record:
            return {"status": "invalid", "record": None}
        if record["attempts"] >= max_attempts:
            return {"status": "capped", "record": record}
        record["attempts"] += 1
        record["status"] = "pending_generation"
        return {"status": "ready", "record": copy.deepcopy(record)}

    def set_relationship_recovery_status(
        self,
        user_id,
        source_id,
        status,
        error=None,
    ):
        record = self.recovery.get((user_id, source_id))
        if not record:
            return False
        record["status"] = status
        record["error"] = error
        if status != "pending_persistence":
            record["expected_edges"] = []
        return True

    def set_pending_relationship_edges(self, user_id, source_id, edges):
        record = self.recovery.get((user_id, source_id))
        if not record:
            return False
        record["status"] = "pending_persistence"
        record["expected_edges"] = copy.deepcopy(edges)
        return True

    def complete_relationship_recovery(self, user_id, source_id):
        record = self.recovery.get((user_id, source_id))
        if not record:
            return {"status": "invalid"}
        for edge in record["expected_edges"]:
            if not self.add_relationship(
                user_id,
                source_id,
                edge["target_id"],
                edge["relationship"],
            ):
                return {"status": "failed"}
        record["status"] = "succeeded"
        record["expected_edges"] = []
        return {"status": "succeeded"}

    def finish_empty_relationship_recovery(self, user_id, source_id):
        return self.set_relationship_recovery_status(
            user_id,
            source_id,
            "succeeded_empty",
        )

    def get_relationship_recovery(self, user_id):
        return {
            "valid": True,
            "records": [
                copy.deepcopy(record)
                for (record_user, _), record in self.recovery.items()
                if record_user == user_id
            ],
            "invalid_records": [],
            "error": None,
        }


def make_agent():
    agent = Agent.__new__(Agent)
    agent.user_id = "user-a"
    agent.graph = FakeGraph()
    return agent


def memory_extraction(text="new fact"):
    return json.dumps(
        {
            "memories": [
                {
                    "text": text,
                    "importance": 8,
                    "confidence": 0.9,
                    "reason": "synthetic test",
                }
            ]
        }
    )


def evolution_response(memory_id="old", text="updated fact"):
    return json.dumps(
        {
            "updates": [
                {
                    "memory_id": memory_id,
                    "text": text,
                    "importance": 9,
                    "confidence": 0.95,
                    "reason": "updated synthetic fact",
                }
            ]
        }
    )


class MemoryLifecycleTests(unittest.TestCase):
    def test_successful_insertion_adds_confirmed_graph_node(self):
        agent = make_agent()
        agent.memory = FakeMemory(add_result={"event_id": "event-1"})
        agent.client = types.SimpleNamespace(
            models=FakeModels([memory_extraction()])
        )
        agent.deduplicate_memory = lambda *args: False
        agent.get_memory_id = lambda event_id: "memory-1"
        relationship_calls = []
        agent.create_memory_relationships = (
            lambda *args: relationship_calls.append(args) or True
        )

        result = agent.remember("synthetic input", "synthetic answer")

        self.assertTrue(result)
        self.assertEqual(agent.graph.nodes[0]["id"], "memory-1")
        self.assertEqual(len(relationship_calls), 1)
        self.assertEqual(agent.memory.deleted, [])

    def test_missing_event_or_memory_id_preserves_added_memory(self):
        cases = (
            ({"status": "accepted"}, None),
            ({"event_id": "event-1"}, None),
        )
        for add_result, memory_id in cases:
            with self.subTest(add_result=add_result):
                agent = make_agent()
                agent.memory = FakeMemory(add_result=add_result)
                agent.client = types.SimpleNamespace(
                    models=FakeModels([memory_extraction()])
                )
                agent.deduplicate_memory = lambda *args: False
                agent.get_memory_id = lambda event_id: memory_id

                result = agent.remember("synthetic input", "answer")

                self.assertTrue(result)
                self.assertEqual(agent.memory.deleted, [])
                self.assertEqual(agent.graph.nodes, [])

    def test_insertion_graph_node_failure_preserves_mem0_memory(self):
        agent = make_agent()
        agent.graph = FakeGraph(node_error=OSError("synthetic graph write error"))
        agent.memory = FakeMemory(add_result={"event_id": "event-1"})
        agent.client = types.SimpleNamespace(
            models=FakeModels([memory_extraction()])
        )
        agent.deduplicate_memory = lambda *args: False
        agent.get_memory_id = lambda event_id: "memory-1"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = agent.remember("synthetic input", "answer")
        self.assertTrue(result)
        self.assertEqual(agent.memory.deleted, [])
        self.assertIn("graph node creation failed", output.getvalue())

    def test_insertion_relationship_failure_is_reported_as_partial(self):
        agent = make_agent()
        agent.memory = FakeMemory(add_result={"event_id": "event-1"})
        agent.client = types.SimpleNamespace(
            models=FakeModels([memory_extraction()])
        )
        agent.deduplicate_memory = lambda *args: False
        agent.get_memory_id = lambda event_id: "memory-1"
        agent.create_memory_relationships = lambda *args: False
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = agent.remember("synthetic input", "answer")
        self.assertTrue(result)
        self.assertIn("relationship persistence failed", output.getvalue())
        self.assertEqual(agent.memory.deleted, [])

    def test_failed_replacement_add_preserves_old_memory(self):
        self._run_update(
            add_error=RuntimeError("synthetic replacement failure"),
            expected_deleted=[],
            expected_old_graph_deleted=[],
            expected_result=False,
        )

    def test_missing_or_unconfirmed_replacement_id_preserves_old_memory(self):
        for add_result, resolved_id in (
            (None, None),
            ({}, None),
            ({"event_id": "event-1"}, None),
        ):
            with self.subTest(add_result=add_result, resolved_id=resolved_id):
                self._run_update(
                    add_result=add_result,
                    resolved_id=resolved_id,
                    expected_deleted=[],
                    expected_old_graph_deleted=[],
                    expected_result=False,
                )

    def test_old_delete_failure_preserves_replacement_and_reports_partial(self):
        output, agent, result = self._run_update(
            add_result={"event_id": "new-event"},
            resolved_id="replacement",
            delete_result={"success": False},
            expected_deleted=["old"],
            expected_old_graph_deleted=[],
            expected_result=True,
        )
        self.assertIn("partially completed", output)
        self.assertIn("replacement", [node["id"] for node in agent.graph.nodes])
        metadata = agent.memory.added[0]["metadata"]
        self.assertEqual(metadata["importance"], 9)
        self.assertEqual(metadata["confidence"], 0.95)
        self.assertEqual(metadata["reason"], "updated synthetic fact")
        saved_evolution = json.loads(metadata["evolution"])
        self.assertEqual(saved_evolution["history"][0]["memory_id"], "older")
        self.assertEqual(saved_evolution["history"][1]["memory_id"], "old")
        self.assertTrue(result)

    def test_old_delete_exception_preserves_replacement_and_reports_partial(self):
        output, agent, result = self._run_update(
            add_result={"event_id": "new-event"},
            resolved_id="replacement",
            delete_result=None,
            expected_deleted=["old"],
            expected_old_graph_deleted=[],
            expected_result=True,
            delete_error=RuntimeError("synthetic delete exception"),
        )
        self.assertIn("partially completed", output)
        self.assertIn("replacement", [node["id"] for node in agent.graph.nodes])
        self.assertTrue(result)

    def test_replacement_creation_precedes_old_delete(self):
        operations = []
        agent = make_agent()
        agent.memory = FakeMemory(
            add_result={"event_id": "new-event"},
            delete_result={"success": True},
        )
        original_add = agent.memory.add
        original_delete = agent.memory.delete

        def add(*args, **kwargs):
            operations.append("add")
            return original_add(*args, **kwargs)

        def delete(memory_id):
            operations.append("delete")
            return original_delete(memory_id)

        agent.memory.add = add
        agent.memory.delete = delete
        agent.graph.nodes = [
            {"user_id": agent.user_id, "id": "old", "memory_text": "old fact"}
        ]
        agent.client = types.SimpleNamespace(
            models=FakeModels([evolution_response()])
        )
        agent.get_memory_id = lambda event_id: "replacement"
        agent.create_memory_relationships = lambda *args: True

        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.update_memory("new information", [self.old_memory()])

        self.assertTrue(result)
        self.assertEqual(operations, ["add", "delete"])
        self.assertEqual(agent.graph.deleted_nodes, [(agent.user_id, "old")])

    def test_deduplication_confirms_replacement_then_deletes_unique_ids(self):
        agent = make_agent()
        agent.memory = FakeMemory(
            add_result={"event_id": "merged-event"},
            delete_result={"success": True},
        )
        operations = []
        original_add = agent.memory.add
        original_delete = agent.memory.delete

        def add(*args, **kwargs):
            operations.append("add")
            return original_add(*args, **kwargs)

        def delete(memory_id):
            operations.append(memory_id)
            return original_delete(memory_id)

        agent.memory.add = add
        agent.memory.delete = delete
        agent.recall = lambda query: {
            "results": [
                {"id": "d1", "memory": "fact one", "score": 0.9},
                {"id": "d2", "memory": "fact two", "score": 0.8},
            ]
        }
        agent.client = types.SimpleNamespace(
            models=FakeModels(
                [
                    json.dumps(
                        {
                            "duplicate_memory_ids": ["d1", "d1", "d2"],
                            "merged_memory": "merged fact",
                            "importance": 8,
                            "confidence": 0.9,
                            "reason": "merged reason",
                        }
                    )
                ]
            )
        )
        agent.get_memory_id = lambda event_id: "merged"
        agent.create_memory_relationships = lambda *args: True

        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.deduplicate_memory("new fact", 7, "reason", 0.8)

        self.assertTrue(result)
        self.assertEqual(operations, ["add", "d1", "d2"])
        self.assertEqual(agent.memory.deleted, ["d1", "d2"])
        metadata = agent.memory.added[0]["metadata"]
        self.assertEqual(metadata["importance"], 8)
        self.assertEqual(metadata["confidence"], 0.9)
        self.assertEqual(metadata["reason"], "merged reason")

    def test_failed_deduplication_delete_preserves_merged_replacement(self):
        agent = self._make_dedup_agent({"success": False})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = agent.deduplicate_memory("new fact", 7, "reason", 0.8)

        self.assertTrue(result)
        self.assertIn("partially completed", output.getvalue())
        self.assertEqual(agent.memory.deleted, ["d1"])
        self.assertEqual(agent.graph.nodes[0]["id"], "merged")

    def test_deduplication_add_failure_never_deletes_original(self):
        agent = self._make_dedup_agent(add_error=RuntimeError("add failed"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.deduplicate_memory("new fact", 7, "reason", 0.8)
        self.assertFalse(result)
        self.assertEqual(agent.memory.deleted, [])

    def test_deduplication_missing_replacement_id_never_deletes_original(self):
        agent = self._make_dedup_agent()
        agent.memory.add_result = {"event_id": "merged-event"}
        agent.get_memory_id = lambda event_id: None
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.deduplicate_memory("new fact", 7, "reason", 0.8)
        self.assertFalse(result)
        self.assertEqual(agent.memory.deleted, [])

    def test_deduplication_preserves_evolution_metadata(self):
        agent = self._make_dedup_agent({"success": True})
        agent.recall = lambda query: {
            "results": [
                {
                    "id": "d1",
                    "memory": "old",
                    "score": 0.9,
                    "metadata": {
                        "importance": 6,
                        "source": "imported",
                        "evolution": {
                            "history": [
                                {"memory_id": "older", "from": "ancient", "to": "old"}
                            ]
                        },
                        "confidence": 0.7,
                        "created_at": "2025-01-01T00:00:00Z",
                    },
                }
            ]
        }
        with contextlib.redirect_stdout(io.StringIO()):
            agent.deduplicate_memory("new fact", 7, "reason", 0.8)
        merged_metadata = agent.memory.added[0]["metadata"]
        self.assertEqual(merged_metadata["source"], "imported")
        self.assertIn("evolution", merged_metadata)
        self.assertEqual(
            json.loads(merged_metadata["evolution"])["history"][0]["memory_id"],
            "older",
        )
        source_record = merged_metadata["source_memories"][0]
        self.assertEqual(source_record["id"], "d1")
        self.assertEqual(source_record["metadata"]["confidence"], 0.7)
        self.assertEqual(
            source_record["metadata"]["created_at"],
            "2025-01-01T00:00:00Z",
        )

    def test_deduplication_deduplicates_shared_history_and_preserves_legacy_metadata(self):
        agent = self._make_dedup_agent({"success": True})
        shared_history = {
            "memory_id": "older",
            "from": "ancient",
            "to": "old",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        agent.recall = lambda query: {
            "results": [
                {
                    "id": "d1",
                    "memory": "old",
                    "score": 0.9,
                    "metadata": {
                        "source": "imported",
                        "evolution": {"history": [shared_history]},
                    },
                },
                {
                    "id": "d2",
                    "memory": "old duplicate",
                    "score": 0.8,
                    "metadata": {
                        "source": "conversation",
                        "evolution": "legacy-unparseable-value",
                    },
                },
            ]
        }
        agent.client = types.SimpleNamespace(
            models=FakeModels(
                [
                    json.dumps(
                        {
                            "duplicate_memory_ids": ["d1", "d2"],
                            "merged_memory": "merged fact",
                        }
                    )
                ]
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.deduplicate_memory("new fact", 7, "reason", 0.8)

        self.assertTrue(result)
        merged_metadata = agent.memory.added[0]["metadata"]
        evolution = json.loads(merged_metadata["evolution"])
        self.assertEqual(evolution["history"], [shared_history])
        self.assertIn(
            {
                "memory_id": "d2",
                "value": "legacy-unparseable-value",
            },
            evolution["source_evolutions"],
        )
        self.assertEqual(
            [item["metadata"]["source"] for item in merged_metadata["source_memories"]],
            ["imported", "conversation"],
        )

    def test_relationship_failure_is_returned_to_callers(self):
        agent = make_agent()
        agent.graph.nodes = [
            {"user_id": agent.user_id, "id": "new", "memory_text": "new"},
            {"user_id": agent.user_id, "id": "target", "memory_text": "target"},
        ]
        agent.graph.relationship_result = False
        agent.client = types.SimpleNamespace(
            models=FakeModels(
                [
                    json.dumps(
                        {
                            "relationships": [
                                {
                                    "target_id": "target",
                                    "relationship": "related_to",
                                }
                            ]
                        }
                    )
                ]
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(agent.create_memory_relationships("new", "new"))

    def test_malformed_relationship_output_is_rejected_before_edge_writes(self):
        for text in (
            "{}",
            '{"relationships": {}}',
            '{"relationships": [null]}',
            '{"relationships": [{"target_id": "missing", "relationship": "knows"}]}',
        ):
            with self.subTest(text=text):
                agent = make_agent()
                agent.graph.nodes = [
                    {"user_id": agent.user_id, "id": "new", "memory_text": "new"},
                    {"user_id": agent.user_id, "id": "target", "memory_text": "target"},
                ]
                agent.client = types.SimpleNamespace(
                    models=FakeModels([text])
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertFalse(
                        agent.create_memory_relationships("new", "new")
                    )
                self.assertEqual(agent.graph.edges, [])

    @staticmethod
    def old_memory():
        return {
            "id": "old",
            "memory": "old fact",
            "metadata": {
                "importance": 5,
                "confidence": 0.6,
                "reason": "prior reason",
                "source": "synthetic",
                "evolution": {
                    "history": [
                        {
                            "memory_id": "older",
                            "from": "older fact",
                            "to": "old fact",
                        }
                    ]
                },
            },
        }

    def _run_update(
        self,
        add_result=None,
        resolved_id=None,
        add_error=None,
        delete_result=None,
        expected_deleted=None,
        expected_old_graph_deleted=None,
        expected_result=None,
        delete_error=None,
    ):
        agent = make_agent()
        agent.memory = FakeMemory(
            add_result=add_result,
            delete_result=delete_result,
            add_error=add_error,
            delete_error=delete_error,
        )
        agent.graph.nodes = [
            {"user_id": agent.user_id, "id": "old", "memory_text": "old fact"}
        ]
        agent.client = types.SimpleNamespace(
            models=FakeModels([evolution_response()])
        )
        agent.get_memory_id = lambda event_id: resolved_id
        agent.create_memory_relationships = lambda *args: True
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = agent.update_memory("new information", [self.old_memory()])
        self.assertEqual(result, expected_result)
        self.assertEqual(agent.memory.deleted, expected_deleted)
        self.assertEqual(agent.graph.deleted_nodes, expected_old_graph_deleted)
        return output.getvalue(), agent, result

    def _make_dedup_agent(self, delete_result=None, add_error=None):
        agent = make_agent()
        agent.memory = FakeMemory(
            add_result={"event_id": "merged-event"},
            delete_result=delete_result,
            add_error=add_error,
        )
        agent.recall = lambda query: {
            "results": [{"id": "d1", "memory": "old", "score": 0.9}]
        }
        agent.client = types.SimpleNamespace(
            models=FakeModels(
                [
                    json.dumps(
                        {
                            "duplicate_memory_ids": ["d1"],
                            "merged_memory": "merged fact",
                        }
                    )
                ]
            )
        )
        agent.get_memory_id = lambda event_id: "merged"
        agent.create_memory_relationships = lambda *args: True
        return agent


class DeleteClassificationTests(unittest.TestCase):
    def test_classifier_known_and_unknown_shapes(self):
        cases = (
            ({"success": True}, "confirmed"),
            ({"deleted": True}, "confirmed"),
            ({"success": False}, "failed"),
            ({"deleted": False}, "failed"),
            ({"success": True, "deleted": False}, "unconfirmed"),
            ({"success": True, "status": "failed"}, "unconfirmed"),
            ({}, "unconfirmed"),
            ({"unexpected": "success"}, "unconfirmed"),
            ({"success": "true"}, "unconfirmed"),
            (None, "unconfirmed"),
            ([], "unconfirmed"),
        )
        for result, expected in cases:
            with self.subTest(result=result):
                self.assertEqual(
                    MemoryManager.delete_result_status(result),
                    expected,
                )

    def test_delete_exception_is_unconfirmed_in_clear(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager.get_all_complete = lambda user_id: {
            "count": 1,
            "results": [{"id": "mem-1", "memory": "synthetic"}],
        }
        manager.delete = lambda memory_id: (_ for _ in ()).throw(
            RuntimeError("synthetic delete error")
        )
        result = manager.clear("user-a")
        self.assertEqual(result["unconfirmed_ids"], ["mem-1"])
        self.assertIn("mem-1", result["remaining_ids"])


class PaginationTests(unittest.TestCase):
    class Client:
        def __init__(self, pages=None, error=None):
            self.pages = pages or {}
            self.error = error
            self.calls = []

        def get_all(self, filters, page):
            self.calls.append((filters, page))
            if self.error:
                raise self.error
            return self.pages[page]

    def manager_for(self, pages=None, error=None):
        manager = MemoryManager.__new__(MemoryManager)
        manager.client = self.Client(pages=pages, error=error)
        return manager

    def test_complete_multi_page_retrieval(self):
        manager = self.manager_for(
            {
                1: {
                    "count": 2,
                    "next": "next-page",
                    "results": [{"id": "one", "memory": "first"}],
                },
                2: {
                    "count": 2,
                    "next": None,
                    "results": [{"id": "two", "memory": "second"}],
                },
            }
        )
        result = manager.get_all_complete("user-a")
        self.assertEqual([item["id"] for item in result["results"]], ["one", "two"])
        self.assertEqual(manager.client.calls[1][0], {"user_id": "user-a"})

    def test_count_mismatch_missing_duplicate_ids_and_invalid_text_rejected(self):
        cases = (
            (
                {
                    1: {
                        "count": 2,
                        "next": "next",
                        "results": [{"id": "one", "memory": "first"}],
                    },
                    2: {
                        "count": 3,
                        "next": None,
                        "results": [{"id": "two", "memory": "second"}],
                    },
                },
                "count mismatch",
            ),
            (
                {
                    1: {
                        "count": 1,
                        "next": None,
                        "results": [{"memory": "no ID"}],
                    }
                },
                "missing ID",
            ),
            (
                {
                    1: {
                        "count": 2,
                        "next": "next",
                        "results": [{"id": "same", "memory": "first"}],
                    },
                    2: {
                        "count": 2,
                        "next": None,
                        "results": [{"id": "same", "memory": "again"}],
                    },
                },
                "duplicate ID",
            ),
            (
                {
                    1: {
                        "count": 1,
                        "next": None,
                        "results": [{"id": "one", "memory": "  "}],
                    }
                },
                "blank memory text",
            ),
            (
                {
                    1: {
                        "count": 1,
                        "next": None,
                        "results": [{"id": "one", "memory": None}],
                    }
                },
                "non-string memory text",
            ),
        )
        for pages, description in cases:
            with self.subTest(description=description):
                with self.assertRaises(ValueError):
                    self.manager_for(pages).get_all_complete("user-a")

    def test_malformed_pagination_metadata_and_retrieval_exception(self):
        for page in (
            {"count": True, "next": None, "results": []},
            {"count": 0, "results": []},
            {"count": 0, "next": 3, "results": []},
        ):
            with self.subTest(page=page):
                with self.assertRaises(ValueError):
                    self.manager_for({1: page}).get_all_complete("user-a")

        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            self.manager_for(error=RuntimeError("retrieval failed")).get_all_complete(
                "user-a"
            )

    def test_clear_retains_recovery_ids_when_final_retrieval_fails(self):
        manager = MemoryManager.__new__(MemoryManager)
        calls = []

        def retrieve(user_id):
            calls.append(user_id)
            if len(calls) == 1:
                return {
                    "count": 1,
                    "results": [{"id": "mem-1", "memory": "synthetic"}],
                }
            raise RuntimeError("final retrieval failed")

        manager.get_all_complete = retrieve
        manager.delete = lambda memory_id: {"success": False}
        result = manager.clear("user-a")
        self.assertFalse(result["remaining_retrieval_complete"])
        self.assertEqual(result["remaining_ids"], ["mem-1"])
        self.assertEqual(result["failed_ids"], ["mem-1"])


class KnowledgeGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "graph.json")
        self.graph = KnowledgeGraph(self.path)

    def test_add_nodes_relationship_duplicate_and_user_isolation(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-a", "a2", "A two")
        self.graph.add_node("user-b", "b1", "B one")
        self.assertTrue(
            self.graph.add_relationship("user-a", "a1", "a2", "related_to")
        )
        self.assertTrue(
            self.graph.add_relationship("user-a", "a1", "a2", "related_to")
        )
        self.assertEqual(
            len(self.graph.get_user_graph("user-a")["relationships"]),
            1,
        )
        self.assertFalse(
            self.graph.add_relationship("user-a", "b1", "a1", "cross_user")
        )
        self.assertEqual(self.graph.get_user_graph("user-b")["relationships"], [])

    def test_missing_invalid_and_cross_user_endpoints_leave_graph_unchanged(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-b", "b1", "B one")
        before = copy.deepcopy(self.graph.graph)
        for args in (
            ("user-a", "missing", "a1", "related"),
            ("user-a", "a1", "missing", "related"),
            ("user-a", " ", "a1", "related"),
            ("user-a", "a1", "a1", ""),
            ("user-a", "b1", "a1", "cross_user"),
        ):
            with self.subTest(args=args):
                self.assertFalse(self.graph.add_relationship(*args))
                self.assertEqual(self.graph.graph, before)

    def test_sync_validates_before_mutation_and_is_user_scoped(self):
        self.graph.add_node("user-a", "old-a", "Old A")
        self.graph.add_node("user-b", "private-b", "Private B")
        self.graph.add_relationship("user-a", "old-a", "old-a", "self")
        before = copy.deepcopy(self.graph.graph)
        with self.assertRaises(ValueError):
            self.graph.sync_user_graph(
                "user-a",
                {"count": 1, "results": [{"id": "new", "memory": ""}]},
            )
        self.assertEqual(self.graph.graph, before)

        self.graph.sync_user_graph(
            "user-a",
            {
                "count": 1,
                "results": [{"id": "new-a", "memory": "New A"}],
            },
        )
        self.assertEqual(
            [node["id"] for node in self.graph.get_user_graph("user-a")["nodes"]],
            ["new-a"],
        )
        self.assertEqual(
            [node["id"] for node in self.graph.get_user_graph("user-b")["nodes"]],
            ["private-b"],
        )

    def test_sync_rejects_id_only_record_without_mutating_graph(self):
        self.graph.add_node("user-a", "old", "Old memory")
        before = copy.deepcopy(self.graph.graph)
        with self.assertRaises(ValueError):
            self.graph.sync_user_graph(
                "user-a",
                {"count": 1, "results": [{"id": "new"}]},
            )
        self.assertEqual(self.graph.graph, before)

    def test_save_failure_propagates_and_is_not_reported_as_persisted(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-a", "a2", "A two")
        with patch.object(self.graph, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.graph.add_relationship("user-a", "a1", "a2", "related")

    def test_relationship_save_failure_rolls_back_in_memory_edge(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-a", "a2", "A two")
        before = copy.deepcopy(self.graph.graph)
        with patch.object(self.graph, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.graph.add_relationship("user-a", "a1", "a2", "related")
        self.assertEqual(self.graph.graph, before)

    def test_node_add_and_update_save_failures_leave_graph_unchanged(self):
        self.graph.add_node("user-a", "a1", "A one")
        before = copy.deepcopy(self.graph.graph)
        for memory_text in ("A two", "A one updated"):
            with self.subTest(memory_text=memory_text):
                with patch.object(self.graph, "save", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        self.graph.add_node("user-a", "a2" if memory_text == "A two" else "a1", memory_text)
                self.assertEqual(self.graph.graph, before)

    def test_node_delete_save_failure_leaves_nodes_and_edges_unchanged(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-a", "a2", "A two")
        self.graph.add_relationship("user-a", "a1", "a2", "related")
        before = copy.deepcopy(self.graph.graph)
        with patch.object(self.graph, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.graph.delete_node("user-a", "a1")
        self.assertEqual(self.graph.graph, before)

    def test_sync_save_failure_leaves_graph_unchanged(self):
        self.graph.add_node("user-a", "a1", "A one")
        before = copy.deepcopy(self.graph.graph)
        memories = {
            "count": 1,
            "results": [{"id": "a2", "memory": "A two"}],
        }
        with patch.object(self.graph, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.graph.sync_user_graph("user-a", memories)
        self.assertEqual(self.graph.graph, before)

    def test_atomic_save_failure_preserves_existing_file_contents(self):
        self.graph.add_node("user-a", "a1", "A one")
        self.graph.add_node("user-a", "a2", "A two")
        with open(self.path, "rb") as graph_file:
            original_contents = graph_file.read()
        before = copy.deepcopy(self.graph.graph)
        with patch("app.graph.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.graph.add_relationship("user-a", "a1", "a2", "related")
        self.assertEqual(self.graph.graph, before)
        with open(self.path, "rb") as graph_file:
            self.assertEqual(graph_file.read(), original_contents)


class RelationshipRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "graph.json")
        self.graph = KnowledgeGraph(self.path)
        self.agent = Agent.__new__(Agent)
        self.agent.user_id = "user-a"
        self.agent.graph = self.graph
        self.agent.memory = types.SimpleNamespace(
            add=self._forbidden_mem0,
            delete=self._forbidden_mem0,
            get_all=self._forbidden_mem0,
            get_all_complete=self._forbidden_mem0,
            search=self._forbidden_mem0,
            clear=self._forbidden_mem0,
        )

    @staticmethod
    def _forbidden_mem0(*args, **kwargs):
        raise AssertionError("Relationship recovery must not call Mem0.")

    @staticmethod
    def relationship_response(target_id="target", relationship="related_to"):
        return json.dumps(
            {
                "relationships": [
                    {
                        "target_id": target_id,
                        "relationship": relationship,
                    }
                ]
            }
        )

    def add_source_and_target(self, source="source", target="target"):
        self.graph.add_node("user-a", source, "Source memory")
        self.graph.add_node("user-a", target, "Target memory")

    @staticmethod
    def pending_record(source_id):
        return {
            "source_id": source_id,
            "status": "pending_generation",
            "attempts": 0,
            "error": None,
            "expected_edges": [],
        }

    def write_recovery_queue(self, size):
        queue = {
            f"source-{index}": self.pending_record(f"source-{index}")
            for index in range(size)
        }
        with open(self.path, "w", encoding="utf-8") as graph_file:
            json.dump(
                {
                    "user-a": {
                        "nodes": [],
                        "relationships": [],
                        "relationship_recovery": queue,
                    }
                },
                graph_file,
            )
        self.graph = KnowledgeGraph(self.path)
        self.agent.graph = self.graph
        return queue

    def test_apply_reports_successful_source_and_outcome(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.agent.client = types.SimpleNamespace(
            models=FakeModels([self.relationship_response()])
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["succeeded"], 1)
        self.assertIn(
            {"source_id": "source", "status": "succeeded"},
            report["details"],
        )

    def test_apply_reports_failed_source_and_reason(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.agent.client = types.SimpleNamespace(
            models=FakeModels([RuntimeError("synthetic generation failure")])
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["failed"], 1)
        failure = next(
            detail for detail in report["details"]
            if detail["source_id"] == "source"
        )
        self.assertEqual(failure["status"], "failed")
        self.assertIn("synthetic generation failure", failure["reason"])

    def test_exactly_200_recovery_records_are_valid(self):
        expected_queue = self.write_recovery_queue(200)
        recovery = self.graph.get_relationship_recovery("user-a")
        self.assertTrue(recovery["valid"])
        self.assertEqual(recovery["queue_size"], 200)
        self.assertEqual(len(recovery["records"]), 200)
        self.assertEqual(len(expected_queue), 200)

    def test_oversized_recovery_queue_is_reported_and_preserved(self):
        expected_queue = self.write_recovery_queue(201)
        before = copy.deepcopy(self.graph.graph)
        recovery = self.graph.get_relationship_recovery("user-a")
        self.assertFalse(recovery["valid"])
        self.assertEqual(recovery["queue_size"], 201)
        self.assertEqual(recovery["blocked_count"], 201)
        self.assertIn("exceeding the limit of 200", recovery["error"])
        self.assertEqual(
            self.graph.graph["user-a"]["relationship_recovery"],
            expected_queue,
        )
        self.assertEqual(self.graph.graph, before)

    def test_oversized_queue_apply_does_not_call_model_or_mutate(self):
        expected_queue = self.write_recovery_queue(201)
        before_graph = copy.deepcopy(self.graph.graph)
        with open(self.path, "rb") as graph_file:
            before_file = graph_file.read()
        models = FakeModels([])
        self.agent.client = types.SimpleNamespace(models=models)
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["blocked"], 201)
        self.assertEqual(report["attempted"], 0)
        self.assertIn("exceeding the limit of 200", report["error"])
        self.assertEqual(models.calls, 0)
        self.assertEqual(self.graph.graph, before_graph)
        self.assertEqual(
            self.graph.graph["user-a"]["relationship_recovery"],
            expected_queue,
        )
        with open(self.path, "rb") as graph_file:
            self.assertEqual(graph_file.read(), before_file)

    def test_nonempty_generation_persists_edges_with_success_status(self):
        self.add_source_and_target()
        self.agent.client = types.SimpleNamespace(
            models=FakeModels([self.relationship_response()])
        )
        saves = []
        original_save = self.graph.save

        def capture_save(graph=None):
            if graph is not None:
                saves.append(copy.deepcopy(graph))
            return original_save(graph)

        with patch.object(self.graph, "save", side_effect=capture_save):
            self.assertTrue(
                self.agent.create_memory_relationships("source", "Source memory")
            )

        final_user_graph = saves[-1]["user-a"]
        self.assertEqual(
            final_user_graph["relationships"],
            [
                {
                    "source": "source",
                    "target": "target",
                    "relationship": "related_to",
                }
            ],
        )
        self.assertEqual(
            final_user_graph["relationship_recovery"]["source"]["status"],
            "succeeded",
        )
        self.assertEqual(
            final_user_graph["relationship_recovery"]["source"]["expected_edges"],
            [],
        )

    def test_empty_generation_is_terminal_and_not_retried(self):
        self.add_source_and_target()
        models = FakeModels(['{"relationships": []}'])
        self.agent.client = types.SimpleNamespace(models=models)
        self.assertTrue(self.agent.create_memory_relationships("source", "Source memory"))
        self.assertEqual(
            self.graph.get_relationship_recovery("user-a")["records"][0]["status"],
            "succeeded_empty",
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["attempted"], 0)
        self.assertEqual(models.calls, 1)

    def test_model_failure_and_malformed_output_remain_retryable(self):
        for response in (
            RuntimeError("temporary model failure"),
            '{"relationships": [null]}',
            "{}",
        ):
            with self.subTest(response=response):
                self.graph = KnowledgeGraph(
                    os.path.join(self.temp_dir.name, f"{len(os.listdir(self.temp_dir.name))}.json")
                )
                self.agent.graph = self.graph
                self.add_source_and_target()
                self.agent.client = types.SimpleNamespace(
                    models=FakeModels([response])
                )
                self.assertFalse(
                    self.agent.create_memory_relationships(
                        "source",
                        "Source memory",
                    )
                )
                record = self.graph.get_relationship_recovery("user-a")["records"][0]
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["attempts"], 1)
                self.assertEqual(record["expected_edges"], [])

    def test_final_graph_save_failure_keeps_durable_pending_edges(self):
        self.add_source_and_target()
        self.assertTrue(self.graph.create_relationship_recovery("user-a", "source"))
        self.graph.begin_relationship_attempt("user-a", "source")
        self.graph.set_pending_relationship_edges(
            "user-a",
            "source",
            [{"target_id": "target", "relationship": "related_to"}],
        )
        with open(self.path, "rb") as graph_file:
            durable_before = graph_file.read()
        before = copy.deepcopy(self.graph.graph)
        with patch("app.graph.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                self.graph.complete_relationship_recovery("user-a", "source")
        self.assertEqual(self.graph.graph, before)
        with open(self.path, "rb") as graph_file:
            self.assertEqual(graph_file.read(), durable_before)
        record = self.graph.get_relationship_recovery("user-a")["records"][0]
        self.assertEqual(record["status"], "pending_persistence")
        self.assertEqual(len(record["expected_edges"]), 1)

    def test_missing_source_is_blocked_without_model_call(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.delete_node("user-a", "source")
        models = FakeModels([])
        self.agent.client = types.SimpleNamespace(models=models)
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["blocked"], 1)
        self.assertEqual(models.calls, 0)
        self.assertEqual(
            self.graph.get_relationship_recovery("user-a")["records"][0]["status"],
            "blocked_missing_endpoint",
        )

    def test_missing_target_is_blocked_without_creating_edge(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.begin_relationship_attempt("user-a", "source")
        self.graph.set_pending_relationship_edges(
            "user-a",
            "source",
            [{"target_id": "target", "relationship": "related_to"}],
        )
        self.graph.delete_node("user-a", "target")
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["blocked"], 1)
        self.assertEqual(
            self.graph.get_user_graph("user-a")["relationships"],
            [],
        )
        self.assertEqual(
            self.graph.get_relationship_recovery("user-a")["records"][0]["status"],
            "blocked_missing_endpoint",
        )

    def test_cross_user_endpoint_is_rejected(self):
        self.graph.add_node("user-a", "source", "Source")
        self.graph.add_node("user-b", "target", "Other user's target")
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.begin_relationship_attempt("user-a", "source")
        record = self.graph.get_user_graph("user-a")[
            "relationship_recovery"
        ]["source"]
        record["status"] = "pending_persistence"
        record["expected_edges"] = [
            {"target_id": "target", "relationship": "related_to"}
        ]
        self.graph.save()
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["blocked"], 1)
        self.assertEqual(
            self.graph.get_relationship_recovery("user-a")["records"][0]["status"],
            "blocked_missing_endpoint",
        )
        self.assertEqual(
            self.graph.get_user_graph("user-a")["relationships"],
            [],
        )
        self.assertEqual(
            self.graph.get_user_graph("user-b")["relationships"],
            [],
        )

    def test_failed_generation_can_be_retried_with_next_bounded_attempt(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.agent.client = types.SimpleNamespace(
            models=FakeModels([RuntimeError("temporary failure")])
        )
        first_report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(first_report["failed"], 1)
        first_record = self.graph.get_relationship_recovery("user-a")["records"][0]
        self.assertEqual(first_record["status"], "failed")
        self.assertEqual(first_record["attempts"], 1)

        self.agent.client.models.responses.append('{"relationships": []}')
        second_report = self.agent.recover_relationships(dry_run=False)
        second_record = self.graph.get_relationship_recovery("user-a")["records"][0]
        self.assertEqual(second_report["succeeded_empty"], 1)
        self.assertEqual(second_record["status"], "succeeded_empty")
        self.assertEqual(second_record["attempts"], 2)

    def test_duplicate_edge_retry_is_idempotent(self):
        self.add_source_and_target()
        self.graph.add_relationship("user-a", "source", "target", "related_to")
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.begin_relationship_attempt("user-a", "source")
        self.graph.set_pending_relationship_edges(
            "user-a",
            "source",
            [{"target_id": "target", "relationship": "related_to"}],
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["succeeded"], 1)
        self.assertEqual(
            len(self.graph.get_user_graph("user-a")["relationships"]),
            1,
        )

    def test_dry_run_has_no_model_graph_or_file_mutations(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        before_graph = copy.deepcopy(self.graph.graph)
        with open(self.path, "rb") as graph_file:
            before_file = graph_file.read()
        models = FakeModels([])
        self.agent.client = types.SimpleNamespace(models=models)
        with patch.object(
            self.graph,
            "save",
            side_effect=AssertionError("dry-run attempted a save"),
        ):
            report = self.agent.recover_relationships(dry_run=True)
        self.assertEqual(models.calls, 0)
        self.assertEqual(self.graph.graph, before_graph)
        self.assertEqual(report["eligible"], ["source"])
        with open(self.path, "rb") as graph_file:
            self.assertEqual(graph_file.read(), before_file)

    def test_attempt_limit_prevents_model_calls_after_restart(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        for _ in range(3):
            self.assertEqual(
                self.graph.begin_relationship_attempt(
                    "user-a",
                    "source",
                    max_attempts=3,
                )["status"],
                "ready",
            )
        self.graph.set_relationship_recovery_status(
            "user-a",
            "source",
            "failed",
            "retryable failure",
        )
        restarted_graph = KnowledgeGraph(self.path)
        self.agent.graph = restarted_graph
        models = FakeModels([])
        self.agent.client = types.SimpleNamespace(models=models)
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["capped"], ["source"])
        self.assertEqual(models.calls, 0)
        self.assertEqual(
            restarted_graph.get_relationship_recovery("user-a")["records"][0]["attempts"],
            3,
        )

    def test_batch_limit_is_enforced(self):
        for index in range(7):
            source = f"source-{index}"
            target = f"target-{index}"
            self.add_source_and_target(source, target)
            self.graph.create_relationship_recovery("user-a", source)
        models = FakeModels(['{"relationships": []}'] * 5)
        self.agent.client = types.SimpleNamespace(models=models)
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["attempted"], 5)
        self.assertEqual(models.calls, 5)
        self.assertEqual(report["remaining"], 2)

    def test_recovery_is_user_scoped_and_does_not_call_mem0(self):
        self.add_source_and_target()
        self.graph.add_node("user-b", "b-source", "B source")
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.create_relationship_recovery("user-b", "b-source")
        before_b = copy.deepcopy(self.graph.graph["user-b"])
        self.agent.client = types.SimpleNamespace(
            models=FakeModels(['{"relationships": []}'])
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["succeeded_empty"], 1)
        self.assertEqual(self.graph.graph["user-b"], before_b)

    def test_sync_preserves_recovery_records_unknown_fields_and_malformed_entries(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        self.graph.graph["user-a"]["custom_legacy"] = {"keep": True}
        self.graph.graph["user-a"]["relationship_recovery"]["broken"] = {
            "status": "unknown",
            "legacy": ["preserve"],
        }
        self.graph.sync_user_graph(
            "user-a",
            {"count": 1, "results": [{"id": "source", "memory": "Source"}]},
        )
        user_graph = self.graph.get_user_graph("user-a")
        self.assertEqual(user_graph["custom_legacy"], {"keep": True})
        self.assertIn("broken", user_graph["relationship_recovery"])
        self.assertEqual(
            user_graph["relationship_recovery"]["source"]["status"],
            "pending_generation",
        )
        report = self.graph.get_relationship_recovery("user-a")
        self.assertEqual(report["invalid_records"], ["broken"])

    def test_queue_capacity_never_evicts_unresolved_records(self):
        self.add_source_and_target("source-1", "target-1")
        self.add_source_and_target("source-2", "target-2")
        self.add_source_and_target("source-3", "target-3")
        with patch.object(KnowledgeGraph, "MAX_RECOVERY_RECORDS", 2):
            self.assertTrue(
                self.graph.create_relationship_recovery("user-a", "source-1")
            )
            self.assertTrue(
                self.graph.create_relationship_recovery("user-a", "source-2")
            )
            self.assertFalse(
                self.graph.create_relationship_recovery("user-a", "source-3")
            )
        records = self.graph.get_relationship_recovery("user-a")["records"]
        self.assertEqual({record["source_id"] for record in records}, {"source-1", "source-2"})

    def test_full_queue_is_reported_without_model_call_or_eviction(self):
        self.add_source_and_target("source-1", "target-1")
        self.add_source_and_target("source-2", "target-2")
        self.add_source_and_target("source-3", "target-3")
        with patch.object(KnowledgeGraph, "MAX_RECOVERY_RECORDS", 2):
            self.graph.create_relationship_recovery("user-a", "source-1")
            self.graph.create_relationship_recovery("user-a", "source-2")
            models = FakeModels(['{"relationships": []}'])
            self.agent.client = types.SimpleNamespace(models=models)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(
                    self.agent.create_memory_relationships(
                        "source-3",
                        "Third source",
                    )
                )
        self.assertIn("queue is full", output.getvalue())
        self.assertEqual(models.calls, 0)
        self.assertEqual(
            {
                record["source_id"]
                for record in self.graph.get_relationship_recovery("user-a")["records"]
            },
            {"source-1", "source-2"},
        )

    def test_failed_initial_tracking_write_does_not_call_model(self):
        self.add_source_and_target()
        models = FakeModels(['{"relationships": []}'])
        self.agent.client = types.SimpleNamespace(models=models)
        with patch.object(self.graph, "save", side_effect=OSError("disk full")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(
                    self.agent.create_memory_relationships(
                        "source",
                        "Source memory",
                    )
                )
        self.assertEqual(models.calls, 0)
        self.assertIn("generation was not started", output.getvalue())
        self.assertNotIn("relationship_recovery", self.graph.graph["user-a"])

    def test_recovery_attempt_increment_is_durable_before_generation(self):
        self.add_source_and_target()
        self.graph.create_relationship_recovery("user-a", "source")
        original_generate = self.agent._generate_relationship_edges

        def check_attempt_before_call(source_id):
            record = self.graph.get_relationship_recovery("user-a")["records"][0]
            self.assertEqual(record["attempts"], 1)
            self.assertEqual(record["status"], "pending_generation")
            return original_generate(source_id)

        self.agent._generate_relationship_edges = check_attempt_before_call
        self.agent.client = types.SimpleNamespace(
            models=FakeModels(['{"relationships": []}'])
        )
        report = self.agent.recover_relationships(dry_run=False)
        self.assertEqual(report["succeeded_empty"], 1)

    def test_malformed_recovery_records_are_reported_and_preserved(self):
        self.add_source_and_target()
        user_graph = self.graph.get_user_graph("user-a")
        user_graph["relationship_recovery"] = ["malformed legacy queue"]
        before = copy.deepcopy(self.graph.graph)
        report = self.agent.recover_relationships(dry_run=True)
        self.assertEqual(report["blocked"], 1)
        self.assertEqual(self.graph.graph, before)


class CliAndSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.session_path = os.path.join(self.temp_dir.name, "session.json")
        self.session_patch = patch.object(main_module, "SESSION_FILE", self.session_path)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)

    def test_load_user_handles_invalid_files_and_values(self):
        self.assertIsNone(load_user())
        invalid = (
            "{bad",
            "[]",
            "null",
            "10",
            '"string"',
            '{"user_id": null}',
            '{"user_id": 10}',
            '{"user_id": "  "}',
        )
        for value in invalid:
            with self.subTest(value=value):
                with open(self.session_path, "w", encoding="utf-8") as file:
                    file.write(value)
                self.assertIsNone(load_user())

        with open(self.session_path, "w", encoding="utf-8") as file:
            json.dump({"user_id": "  user-a  "}, file)
        self.assertEqual(load_user(), "user-a")

    def test_relationship_recovery_cli_defaults_to_dry_run(self):
        calls = []

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.memory = types.SimpleNamespace()
                self.graph = types.SimpleNamespace()

            def recover_relationships(self, dry_run=True):
                calls.append(dry_run)
                return {
                    "dry_run": dry_run,
                    "batch_limit": 5,
                    "attempted": 1 if not dry_run else 0,
                    "succeeded": 1 if not dry_run else 0,
                    "succeeded_empty": 0,
                    "pending": 0,
                    "failed": 0,
                    "blocked": 0,
                    "capped": [],
                    "eligible": ["source"],
                    "invalid_records": [],
                    "remaining": 0,
                    "details": [],
                }

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ):
            commands = iter(
                ["/recover-relationships", "/recover-relationships --apply", "exit"]
            )
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()

        self.assertEqual(calls, [True, False])
        self.assertIn("No changes were made", output.getvalue())
        self.assertIn("Relationship recovery results", output.getvalue())

    def test_save_rejects_blank_ids_and_persistence_failure_preserves_file(self):
        for value in ("", "  ", None, 2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    save_user(value)

        save_user("previous")
        with open(self.session_path, "rb") as file:
            original = file.read()
        with patch.object(os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                save_user("next")
        with open(self.session_path, "rb") as file:
            self.assertEqual(file.read(), original)

    def test_startup_rejects_blank_ids_before_saving(self):
        instances = []

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.memory = types.SimpleNamespace()
                self.graph = types.SimpleNamespace()
                instances.append(self)

        commands = iter(["", "   ", "  startup-user  ", "exit"])
        output = io.StringIO()
        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value=None
        ), patch(
            "builtins.input",
            side_effect=lambda prompt="": next(commands),
        ), contextlib.redirect_stdout(output):
            main()

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].user_id, "startup-user")
        self.assertEqual(load_user(), "startup-user")
        self.assertEqual(output.getvalue().count("User ID cannot be empty"), 2)

    def test_login_logout_and_failed_transitions(self):
        instances = []

        class FakeAgent:
            def __init__(self, user_id):
                if user_id == "broken":
                    raise RuntimeError("synthetic agent failure")
                self.user_id = user_id
                self.memory = types.SimpleNamespace()
                self.graph = types.SimpleNamespace()
                instances.append(self)

        save_user("prior")
        agent_patch = patch.object(main_module, "Agent", FakeAgent)
        agent_patch.start()
        self.addCleanup(agent_patch.stop)

        commands = iter(
            [
                "/login    ",
                "/login broken",
                "/user",
                "/logout",
                "   ",
                "/user",
                "/login next-user",
                "/user",
                "/logout",
                " last-user ",
                "/user",
                "exit",
            ]
        )
        output = io.StringIO()
        with patch("builtins.input", side_effect=lambda prompt="": next(commands)):
            with contextlib.redirect_stdout(output):
                main()

        self.assertIn("Current user: prior", output.getvalue())
        self.assertIn("Logout cancelled", output.getvalue())
        self.assertIn("Current user: next-user", output.getvalue())
        self.assertIn("Current user: last-user", output.getvalue())
        self.assertEqual(load_user(), "last-user")
        self.assertEqual(instances[-1].user_id, "last-user")

    def test_failed_login_construction_and_persistence_keep_previous_agent(self):
        instances = []

        class FakeAgent:
            def __init__(self, user_id):
                if user_id == "broken":
                    raise RuntimeError("initialization failed")
                self.user_id = user_id
                self.memory = types.SimpleNamespace()
                self.graph = types.SimpleNamespace()
                instances.append(self)

        save_user("prior")
        with patch.object(main_module, "Agent", FakeAgent):
            commands = iter(["/login broken", "/user", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ):
                with contextlib.redirect_stdout(output):
                    main()
            self.assertEqual(load_user(), "prior")
            self.assertIn("Current user: prior", output.getvalue())

            commands = iter(["/logout", "broken", "/user", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ):
                with contextlib.redirect_stdout(output):
                    main()
            self.assertEqual(load_user(), "prior")
            self.assertIn("Current user: prior", output.getvalue())

            commands = iter(["/login next", "/user", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), patch.object(os, "replace", side_effect=OSError("persistence failed")):
                with contextlib.redirect_stdout(output):
                    main()
            self.assertEqual(load_user(), "prior")
            self.assertIn("Current user: prior", output.getvalue())

            commands = iter(["/logout", "next", "/user", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), patch.object(os, "replace", side_effect=OSError("persistence failed")):
                with contextlib.redirect_stdout(output):
                    main()
            self.assertEqual(load_user(), "prior")
            self.assertIn("Current user: prior", output.getvalue())

    def test_forget_ambiguous_delete_does_not_mutate_graph(self):
        graph_instances = []

        class FakeGraph:
            def __init__(self):
                self.deleted = []
                graph_instances.append(self)

            def delete_node(self, *args):
                self.deleted.append(args)

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = types.SimpleNamespace(
                    delete=lambda memory_id: {
                        "success": True,
                        "deleted": False,
                    },
                    delete_result_status=MemoryManager.delete_result_status,
                )

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ), patch.object(
            main_module,
            "get_active_memories",
            return_value=[{"id": "mem-1", "memory": "synthetic"}],
        ):
            commands = iter(["/forget 1", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ):
                with contextlib.redirect_stdout(output):
                    main()

        self.assertEqual(graph_instances[0].deleted, [])
        self.assertIn("could not be confirmed", output.getvalue().lower())

    def test_forget_confirmed_deletion_updates_graph(self):
        graphs = []

        class FakeGraph:
            def __init__(self):
                self.deleted = []
                graphs.append(self)

            def delete_node(self, *args):
                self.deleted.append(args)

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = types.SimpleNamespace(
                    delete=lambda memory_id: {"success": True},
                    delete_result_status=MemoryManager.delete_result_status,
                )

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ), patch.object(
            main_module,
            "get_active_memories",
            return_value=[{"id": "mem-1", "memory": "synthetic"}],
        ):
            commands = iter(["/forget 1", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()

        self.assertEqual(graphs[0].deleted, [("user-a", "mem-1")])
        self.assertIn("Memory deleted.", output.getvalue())

    def test_forget_negative_and_exception_leave_graph_unchanged(self):
        responses = (
            {"success": False},
            RuntimeError("delete exception"),
        )
        for response in responses:
            with self.subTest(response=response):
                graphs = []

                class FakeGraph:
                    def __init__(self):
                        self.deleted = []
                        graphs.append(self)

                    def delete_node(self, *args):
                        self.deleted.append(args)

                def delete(memory_id):
                    if isinstance(response, Exception):
                        raise response
                    return response

                class FakeAgent:
                    def __init__(self, user_id):
                        self.user_id = user_id
                        self.graph = FakeGraph()
                        self.memory = types.SimpleNamespace(
                            delete=delete,
                            delete_result_status=MemoryManager.delete_result_status,
                        )

                with patch.object(main_module, "Agent", FakeAgent), patch.object(
                    main_module, "load_user", return_value="user-a"
                ), patch.object(
                    main_module,
                    "get_active_memories",
                    return_value=[{"id": "mem-1", "memory": "synthetic"}],
                ):
                    commands = iter(["/forget 1", "exit"])
                    output = io.StringIO()
                    with patch(
                        "builtins.input",
                        side_effect=lambda prompt="": next(commands),
                    ), contextlib.redirect_stdout(output):
                        main()

                self.assertEqual(graphs[0].deleted, [])
                self.assertNotIn("Memory deleted.", output.getvalue())

    def test_clear_partial_and_sync_failure_report_recovery_ids(self):
        graph_instances = []

        class FakeGraph:
            def __init__(self):
                self.sync_calls = []
                graph_instances.append(self)

            def sync_user_graph(self, user_id, memories):
                self.sync_calls.append((user_id, memories))
                raise OSError("graph unavailable")

        result = {
            "retrieval_complete": True,
            "deleted_ids": [],
            "failed_ids": ["mem-1"],
            "unconfirmed_ids": [],
            "remaining_ids": ["mem-1"],
            "remaining_retrieval_complete": True,
            "remaining_memories": {
                "count": 1,
                "results": [{"id": "mem-1", "memory": "synthetic"}],
            },
            "complete": False,
            "error": None,
        }

        class FakeMemory:
            def get_all_complete(self, user_id):
                return {
                    "count": 1,
                    "results": [{"id": "mem-1", "memory": "synthetic"}],
                }

            def clear(self, user_id):
                return result

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = FakeMemory()

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ):
            commands = iter(["/clear", "yes", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()

        self.assertIn("Failed IDs: ['mem-1']", output.getvalue())
        self.assertIn("Remaining IDs: ['mem-1']", output.getvalue())
        self.assertNotIn("All memories were cleared", output.getvalue())
        self.assertEqual(len(graph_instances[0].sync_calls), 1)

    def test_clear_complete_result_reports_success_only_after_sync(self):
        graphs = []

        class FakeGraph:
            def __init__(self):
                self.sync_calls = []
                graphs.append(self)

            def sync_user_graph(self, user_id, memories):
                self.sync_calls.append((user_id, memories))

        class FakeMemory:
            def get_all_complete(self, user_id):
                return {
                    "count": 1,
                    "results": [{"id": "mem-1", "memory": "synthetic"}],
                }

            def clear(self, user_id):
                return {
                    "retrieval_complete": True,
                    "deleted_ids": ["mem-1"],
                    "failed_ids": [],
                    "unconfirmed_ids": [],
                    "remaining_ids": [],
                    "remaining_retrieval_complete": True,
                    "remaining_memories": {"count": 0, "results": []},
                    "complete": True,
                    "error": None,
                }

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = FakeMemory()

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ):
            commands = iter(["/clear", "yes", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()

        self.assertIn("All memories were cleared", output.getvalue())
        self.assertEqual(
            graphs[0].sync_calls,
            [("user-a", {"count": 0, "results": []})],
        )

    def test_clear_initial_retrieval_failure_does_not_sync(self):
        graphs = []

        class FakeGraph:
            def __init__(self):
                self.sync_calls = []
                graphs.append(self)

            def sync_user_graph(self, *args):
                self.sync_calls.append(args)

        class FakeMemory:
            def get_all_complete(self, user_id):
                raise RuntimeError("retrieval failed")

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = FakeMemory()

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ):
            commands = iter(["/clear", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()

        self.assertEqual(graphs[0].sync_calls, [])
        self.assertIn("retrieval failed", output.getvalue())

    def test_graph_command_renders_nodes_and_relationship(self):
        class FakeGraph:
            def sync_user_graph(self, user_id, memories):
                self.user_graph = {
                    "nodes": [
                        {"id": "one", "memory_text": "First"},
                        {"id": "two", "memory_text": "Second"},
                    ],
                    "relationships": [
                        {
                            "source": "one",
                            "target": "two",
                            "relationship": "related_to",
                        }
                    ],
                }

            def get_user_graph(self, user_id):
                return self.user_graph

        class FakeMemory:
            def get_all_complete(self, user_id):
                return {
                    "count": 2,
                    "results": [
                        {"id": "one", "memory": "First"},
                        {"id": "two", "memory": "Second"},
                    ],
                }

        class FakeAgent:
            def __init__(self, user_id):
                self.user_id = user_id
                self.graph = FakeGraph()
                self.memory = FakeMemory()

        with patch.object(main_module, "Agent", FakeAgent), patch.object(
            main_module, "load_user", return_value="user-a"
        ):
            commands = iter(["/graph", "exit"])
            output = io.StringIO()
            with patch(
                "builtins.input",
                side_effect=lambda prompt="": next(commands),
            ), contextlib.redirect_stdout(output):
                main()
        self.assertIn("1 --related_to--> 2", output.getvalue())


class DependencyTests(unittest.TestCase):
    def test_direct_google_genai_import_is_declared(self):
        with open("app/agent.py", encoding="utf-8") as file:
            source = file.read()
        with open("requirements.txt", encoding="utf-8") as file:
            requirements = [line.strip().lower() for line in file]
        self.assertIn("from google import genai", source)
        self.assertTrue(
            any(line.startswith("google-genai") for line in requirements)
        )


if __name__ == "__main__":
    unittest.main()
