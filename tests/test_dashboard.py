import json
import os
import tempfile
import unittest

from app.dashboard import DashboardDataError, load_dashboard_snapshot


class DashboardSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "graph.json")

    def write_graph(self, graph):
        with open(self.path, "w", encoding="utf-8") as graph_file:
            json.dump(graph, graph_file)

    def test_user_scoping_and_recovery_status_counts(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [
                        {
                            "id": "a1",
                            "memory_text": "private memory " * 20,
                            "created_at": "2026-10-23T00:00:00Z",
                        },
                        {"id": "a2", "memory_text": "another private memory"},
                    ],
                    "relationships": [
                        {
                            "source": "a1",
                            "target": "a2",
                            "relationship": "related_to",
                        }
                    ],
                    "relationship_recovery": {
                        "a1": {
                            "source_id": "a1",
                            "status": "pending_generation",
                            "attempts": 1,
                            "error": None,
                            "expected_edges": [],
                        },
                        "a2": {
                            "source_id": "a2",
                            "status": "failed",
                            "attempts": 2,
                            "error": "retry needed",
                            "expected_edges": [],
                        },
                    },
                },
                "user-b": {
                    "nodes": [{"id": "b1", "memory_text": "other user"}],
                    "relationships": [],
                    "relationship_recovery": {
                        "b1": {
                            "source_id": "b1",
                            "status": "blocked_missing_endpoint",
                            "attempts": 0,
                            "error": "missing",
                            "expected_edges": [],
                        }
                    },
                },
            }
        )

        snapshot = load_dashboard_snapshot(self.path, " user-a ")
        self.assertEqual(snapshot["user_id"], "user-a")
        self.assertEqual(snapshot["node_count"], 2)
        self.assertEqual(snapshot["relationship_count"], 1)
        self.assertEqual(snapshot["pending_recovery_count"], 1)
        self.assertEqual(snapshot["failed_or_blocked_count"], 1)
        self.assertEqual(
            {record["source_id"] for record in snapshot["recovery_records"]},
            {"a1", "a2"},
        )
        self.assertEqual(snapshot["memories"][0]["node_id"], "a1")
        self.assertLessEqual(len(snapshot["memories"][0]["preview"]), 120)
        self.assertEqual(
            snapshot["memories"][0]["created_at"],
            "2026-10-23T00:00:00Z",
        )
        self.assertEqual(snapshot["memories"][0]["updated_at"], "—")
        self.assertEqual(snapshot["memories"][1]["created_at"], "—")
        self.assertEqual(snapshot["memories"][1]["updated_at"], "—")
        self.assertEqual(
            snapshot["relationships"][0],
            {
                "source_id": "a1",
                "relationship": "related_to",
                "target_id": "a2",
            },
        )
        self.assertNotIn("user-b", repr(snapshot))
        self.assertNotIn("other user", repr(snapshot))
        self.assertNotIn("private memory " * 20, repr(snapshot))

    def test_memories_without_text_and_malformed_edges_are_safe(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [{"id": "a1"}],
                    "relationships": [
                        {"source": "missing", "relationship": None},
                        "malformed edge",
                    ],
                }
            }
        )
        snapshot = load_dashboard_snapshot(self.path, "user-a")
        self.assertEqual(
            snapshot["memories"][0]["preview"],
            "No text available",
        )
        self.assertEqual(
            snapshot["relationships"][0]["source_id"],
            "[missing or invalid]",
        )
        self.assertEqual(
            snapshot["relationships"][0]["relationship"],
            "[missing or invalid]",
        )
        self.assertEqual(
            snapshot["relationships"][1]["relationship"],
            "[invalid relationship]",
        )

    def test_stored_memory_preview_redacts_secret_patterns(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [
                        {
                            "id": "a1",
                            "memory_text": (
                                "Project key: MEM0_API_KEY=secret-token "
                                "AIza123456789012345678901234567890"
                            ),
                        }
                    ],
                    "relationships": [],
                }
            }
        )
        preview = load_dashboard_snapshot(self.path, "user-a")["memories"][0][
            "preview"
        ]
        self.assertNotIn("secret-token", preview)
        self.assertNotIn("AIza123456789012345678901234567890", preview)

    def test_empty_graph_file_and_missing_user_have_empty_state(self):
        self.write_graph({})
        snapshot = load_dashboard_snapshot(self.path, "user-a")
        self.assertTrue(snapshot["empty"])
        self.assertEqual(snapshot["node_count"], 0)
        self.assertEqual(snapshot["relationship_count"], 0)

        os.remove(self.path)
        missing = load_dashboard_snapshot(self.path, "user-a")
        self.assertTrue(missing["empty"])

    def test_malformed_graph_file_and_data_raise_clear_errors(self):
        with open(self.path, "w", encoding="utf-8") as graph_file:
            graph_file.write("{bad json")
        with self.assertRaisesRegex(DashboardDataError, "Could not read"):
            load_dashboard_snapshot(self.path, "user-a")

        self.write_graph({"user-a": {"nodes": "invalid", "relationships": []}})
        with self.assertRaisesRegex(DashboardDataError, "malformed"):
            load_dashboard_snapshot(self.path, "user-a")

        self.write_graph([])
        with self.assertRaisesRegex(DashboardDataError, "user-keyed"):
            load_dashboard_snapshot(self.path, "user-a")

    def test_malformed_recovery_record_is_counted_as_blocked(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [],
                    "relationships": [],
                    "relationship_recovery": {"bad": {"status": "unknown"}},
                }
            }
        )
        snapshot = load_dashboard_snapshot(self.path, "user-a")
        self.assertEqual(snapshot["failed_or_blocked_count"], 1)
        self.assertEqual(snapshot["recovery_records"][0]["status"], "invalid")

    def test_recovery_error_does_not_expose_memory_text_or_api_key(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [{"id": "a1", "memory_text": "secret full memory"}],
                    "relationships": [],
                    "relationship_recovery": {
                        "a1": {
                            "source_id": "a1",
                            "status": "failed",
                            "attempts": 1,
                            "error": (
                                "failed for secret full memory api_key=secret-token "
                                "and AIza123456789012345678901234567890"
                            ),
                            "expected_edges": [],
                        }
                    },
                }
            }
        )
        snapshot = load_dashboard_snapshot(self.path, "user-a")
        rendered_error = snapshot["recovery_records"][0]["error"]
        self.assertNotIn("secret full memory", rendered_error)
        self.assertNotIn("secret-token", rendered_error)
        self.assertNotIn("AIza123456789012345678901234567890", rendered_error)
