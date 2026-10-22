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

    def generate_content(self, **kwargs):
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
    def __init__(self, node_error=None, relationship_result=True):
        self.nodes = []
        self.edges = []
        self.deleted_nodes = []
        self.node_error = node_error
        self.relationship_result = relationship_result
        self.sync_calls = []

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
