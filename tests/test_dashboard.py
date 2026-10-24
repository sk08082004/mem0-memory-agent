import json
import os
import tempfile
import unittest

from app.dashboard import DashboardDataError, load_dashboard_snapshot
from app.dashboard_data import (
    JsonGraphSnapshotProvider,
    build_graph_mermaid,
    export_memories_csv,
    filter_memory_items,
    graph_subset,
    normalize_dashboard_snapshot,
    paginate,
)
from app.dashboard import (
    _connected_memories,
    _format_timestamp,
    _integrity_rows,
    _memory_rows,
    _recovery_rows,
    _status_label,
)


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
        self.assertLessEqual(len(snapshot["memories"][0]["preview"]), 120)

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

    def test_empty_snapshot_has_integrity_and_empty_aggregates(self):
        snapshot = normalize_dashboard_snapshot(None, "user-a")

        self.assertTrue(snapshot["empty"])
        self.assertEqual(snapshot["node_count"], 0)
        self.assertEqual(snapshot["relationship_count"], 0)
        self.assertEqual(snapshot["recovery_record_count"], 0)
        self.assertEqual(snapshot["integrity"]["error_count"], 0)
        self.assertEqual(snapshot["eligible_source_ids"], [])

    def test_integrity_checks_duplicates_missing_endpoints_and_malformed_items(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {"id": "same", "memory_text": "A"},
                    {"id": "same", "memory_text": "B"},
                    None,
                    {"id": "isolated"},
                ],
                "relationships": [
                    {
                        "source": "same",
                        "target": "missing",
                        "relationship": "relates",
                    },
                    {
                        "source": "same",
                        "target": "missing",
                        "relationship": "relates",
                    },
                    None,
                ],
            },
            "user-a",
        )
        issue_codes = {
            issue["code"] for issue in snapshot["integrity"]["issues"]
        }

        self.assertIn("duplicate_node_ids", issue_codes)
        self.assertIn("duplicate_relationships", issue_codes)
        self.assertIn("missing_relationship_endpoints", issue_codes)
        self.assertIn("malformed_nodes", issue_codes)
        self.assertIn("malformed_relationships", issue_codes)
        self.assertEqual(snapshot["isolated_node_count"], 3)

    def test_recovery_aggregation_and_eligibility_use_shared_logic(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {"id": "pending", "memory_text": "Pending"},
                    {"id": "failed", "memory_text": "Failed"},
                    {"id": "success", "memory_text": "Success"},
                    {"id": "capped", "memory_text": "Capped"},
                    {"id": "blocked", "memory_text": "Blocked"},
                    {"id": "missing-target", "memory_text": "Missing target"},
                ],
                "relationships": [],
                "relationship_recovery": {
                    "pending": {
                        "source_id": "pending",
                        "status": "pending_generation",
                        "attempts": 0,
                        "error": None,
                        "expected_edges": [],
                    },
                    "failed": {
                        "source_id": "failed",
                        "status": "failed",
                        "attempts": 1,
                        "error": "retry",
                        "expected_edges": [],
                    },
                    "success": {
                        "source_id": "success",
                        "status": "succeeded_empty",
                        "attempts": 1,
                        "error": None,
                        "expected_edges": [],
                    },
                    "capped": {
                        "source_id": "capped",
                        "status": "failed",
                        "attempts": 3,
                        "error": "capped",
                        "expected_edges": [],
                    },
                    "blocked": {
                        "source_id": "blocked",
                        "status": "blocked_missing_endpoint",
                        "attempts": 1,
                        "error": "missing",
                        "expected_edges": [],
                    },
                    "missing-target": {
                        "source_id": "missing-target",
                        "status": "pending_persistence",
                        "attempts": 1,
                        "error": None,
                        "expected_edges": [
                            {
                                "target_id": "absent",
                                "relationship": "related",
                            }
                        ],
                    },
                    "bad": {"status": "unknown"},
                },
            },
            "user-a",
        )
        records = {
            record.get("source_id"): record
            for record in snapshot["recovery_records"]
        }

        self.assertEqual(snapshot["pending_recovery_count"], 2)
        self.assertEqual(snapshot["failed_recovery_count"], 2)
        self.assertEqual(snapshot["succeeded_recovery_count"], 1)
        self.assertEqual(snapshot["blocked_recovery_count"], 1)
        self.assertTrue(records["pending"]["eligible"])
        self.assertTrue(records["failed"]["eligible"])
        self.assertEqual(
            records["capped"]["eligibility_reason"],
            "attempt_cap_reached",
        )
        self.assertEqual(
            records["missing-target"]["eligibility_reason"],
            "target_node_missing",
        )
        self.assertFalse(records["bad"]["valid"])
        self.assertEqual(
            snapshot["memory_items"][0]["recovery"]["status"],
            "pending_generation",
        )

    def test_search_filters_sorting_and_pagination(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {
                        "id": "a",
                        "memory_text": "Alpha project",
                        "created_at": "2026-10-01T00:00:00Z",
                        "updated_at": "2026-10-02T00:00:00Z",
                    },
                    {
                        "id": "b",
                        "memory_text": "Beta project",
                        "created_at": "2026-10-03T00:00:00Z",
                    },
                    {"id": "c", "memory_text": "Gamma"},
                ],
                "relationships": [
                    {"source": "a", "target": "b", "relationship": "knows"}
                ],
                "relationship_recovery": {
                    "a": {
                        "source_id": "a",
                        "status": "failed",
                        "attempts": 1,
                        "error": None,
                        "expected_edges": [],
                    }
                },
            },
            "user-a",
        )
        matches = filter_memory_items(
            snapshot["memory_items"],
            query="PROJECT",
            recovery_statuses=["failed"],
            connectivity="connected",
            sort_by="Most recently created",
        )
        page, pages = paginate(matches, 1, 1)

        self.assertEqual([item["id"] for item in matches], ["a"])
        self.assertEqual([item["id"] for item in page], ["a"])
        self.assertEqual(pages, 1)
        self.assertEqual(snapshot["timeline"], [
            {"date": "2026-10-01", "memories": 1},
            {"date": "2026-10-03", "memories": 1},
        ])

    def test_export_excludes_text_by_default_and_redacts_when_opted_in(self):
        memory = {
            "id": "memory-id",
            "memory_text": "Account api_key=secret-token",
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": None,
            "recovery": {"status": "failed", "attempts": 1},
            "relationship_count": 2,
            "connectivity": "connected",
        }
        default_csv = export_memories_csv([memory])
        opt_in_csv = export_memories_csv([memory], include_memory_text=True)

        self.assertNotIn("memory_text", default_csv.splitlines()[0])
        self.assertNotIn("Account", default_csv)
        self.assertIn("memory_text", opt_in_csv.splitlines()[0])
        self.assertIn("api_key=[redacted]", opt_in_csv)
        self.assertNotIn("secret-token", opt_in_csv)

    def test_graph_visualization_is_bounded_and_marks_isolated_nodes(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {"id": "a", "memory_text": "A meaningful memory description"},
                    {"id": "b", "memory_text": "B meaningful memory description"},
                    {"id": "c", "memory_text": "C meaningful memory description"},
                ],
                "relationships": [
                    {"source": "a", "target": "b", "relationship": "knows"},
                    {"source": "b", "target": "c", "relationship": "works"},
                ],
            },
            "user-a",
        )
        nodes, edges = graph_subset(
            snapshot["memory_items"],
            snapshot["relationship_items"],
            relationship_types=["knows"],
            selected_node_id="a",
            max_nodes=2,
        )
        diagram = build_graph_mermaid(nodes, edges, selected_node_id="a")

        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        self.assertIn('n0["A meaningful memory<br/>description"]:::selected', diagram)
        self.assertIn("|knows|", diagram)
        self.assertNotIn('n0["a"]', diagram)
        self.assertEqual(
            graph_subset(
                snapshot["memory_items"],
                snapshot["relationship_items"],
                selected_node_id="c",
                max_nodes=1,
            )[1],
            [],
        )

    def test_graph_labels_use_memory_text_truncate_and_disambiguate_duplicates(self):
        long_text = (
            "User's project RECOVERY-CHECK-1023 is a Python project that "
            "uses PostgreSQL and stores durable memory."
        )
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {"id": "id-abcdef123", "memory_text": long_text},
                    {"id": "id-123456789", "memory_text": long_text},
                    {"id": "empty", "memory_text": None},
                ],
                "relationships": [],
            },
            "user-a",
        )
        first, second, missing = snapshot["memory_items"]
        diagram = build_graph_mermaid(
            snapshot["memory_items"],
            snapshot["relationship_items"],
        )

        self.assertLessEqual(len(first["graph_preview"]), 54)
        self.assertTrue(first["graph_preview"].endswith("…"))
        self.assertIn("RECOVERY-CHECK-1023", first["graph_preview"])
        self.assertNotEqual(first["graph_label"], second["graph_label"])
        self.assertIn("def123", first["graph_label"])
        self.assertIn("456789", second["graph_label"])
        self.assertEqual(missing["graph_preview"], "Memory text unavailable")
        self.assertIn("RECOVERY-CHECK-1023", diagram)
        self.assertIn("def123", diagram)
        self.assertNotIn(first["id"], diagram)
        self.assertEqual(first["id"], "id-abcdef123")

    def test_memory_rows_prioritize_preview_and_keep_id_secondary(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {
                        "id": "full-uuid-123",
                        "memory_text": "This memory explains the project.",
                        "created_at": "2026-10-01T10:00:00Z",
                    }
                ],
                "relationships": [],
            },
            "user-a",
        )

        rows = _memory_rows(snapshot["memory_items"])

        self.assertEqual(list(rows[0])[0], "Memory")
        self.assertEqual(rows[0]["Memory"], "This memory explains the project.")
        self.assertEqual(rows[0]["Memory ID"], "full-uuid-123")
        self.assertIn("Last updated", rows[0])
        self.assertEqual(_format_timestamp(None), "Unavailable")

    def test_connected_memory_details_use_previews_and_show_missing_endpoints(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [
                    {"id": "source", "memory_text": "Source project memory"},
                    {"id": "target", "memory_text": "Target meeting memory"},
                ],
                "relationships": [
                    {
                        "source": "source",
                        "target": "target",
                        "relationship": "works_with",
                    },
                    {
                        "source": "source",
                        "target": "missing",
                        "relationship": "depends_on",
                    },
                ],
            },
            "user-a",
        )

        related = _connected_memories(snapshot, "source")

        self.assertEqual(len(related), 2)
        self.assertEqual(related[0]["preview"], "Target meeting memory")
        self.assertEqual(related[0]["relationship"], "works_with")
        self.assertEqual(related[0]["direction"], "Outgoing")
        self.assertEqual(related[0]["memory_id"], "target")
        self.assertTrue(related[1]["missing_endpoint"])
        self.assertEqual(related[1]["preview"], "Missing memory endpoint")

    def test_recovery_display_labels_preserve_stored_status_values(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [{"id": "memory", "memory_text": "Project memory"}],
                "relationships": [],
                "relationship_recovery": {
                    "memory": {
                        "source_id": "memory",
                        "status": "succeeded_empty",
                        "attempts": 1,
                        "error": None,
                        "expected_edges": [],
                    }
                },
            },
            "user-a",
        )
        raw_status = snapshot["recovery_records"][0]["status"]
        rows = _recovery_rows(snapshot)

        self.assertEqual(
            _status_label(raw_status),
            "Completed; no new relationships found",
        )
        self.assertEqual(rows[0]["Status"], "Completed; no new relationships found")
        self.assertEqual(rows[0]["Technical status"], "succeeded_empty")
        self.assertEqual(snapshot["recovery_records"][0]["status"], "succeeded_empty")

    def test_integrity_rows_explain_timestamp_gaps_and_next_steps(self):
        snapshot = normalize_dashboard_snapshot(
            {
                "nodes": [{"id": "without-date", "memory_text": "No timestamp"}],
                "relationships": [],
            },
            "user-a",
        )
        rows = _integrity_rows(snapshot["integrity"]["issues"])
        timestamp_row = next(
            row for row in rows
            if "creation" in row["Issue"].casefold()
        )

        self.assertEqual(timestamp_row["Severity"], "Warning")
        self.assertIn("timeline", timestamp_row["Impact"])
        self.assertIn("creation timestamps", timestamp_row["Recommended next step"])
        self.assertEqual(
            snapshot["nodes_missing_created_at_count"],
            1,
        )

    def test_provider_is_read_only_and_scoped_to_selected_user(self):
        self.write_graph(
            {
                "user-a": {
                    "nodes": [{"id": "a", "memory_text": "private A"}],
                    "relationships": [],
                },
                "user-b": {
                    "nodes": [{"id": "b", "memory_text": "private B"}],
                    "relationships": [],
                },
            }
        )
        with open(self.path, "rb") as graph_file:
            before = graph_file.read()

        snapshot = JsonGraphSnapshotProvider(self.path).load_snapshot("user-a")

        self.assertEqual([node["id"] for node in snapshot["memory_items"]], ["a"])
        self.assertNotIn("private B", repr(snapshot))
        with open(self.path, "rb") as graph_file:
            self.assertEqual(graph_file.read(), before)
