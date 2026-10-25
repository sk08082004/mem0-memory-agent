import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.conversation import (
    GENERIC_AGENT_ERROR,
    normalize_user_conversations,
    process_conversation_message,
    start_new_conversation,
)
from app.conversation_store import (
    ConversationStoreError,
    JsonConversationRepository,
)
from app.dashboard import (
    DASHBOARD_SECTIONS,
    DEFAULT_DASHBOARD_SECTION,
    _merge_unsaved_conversations,
    _recent_conversations,
    _render_conversation,
    _render_conversation_message,
    _stash_unsaved_conversation,
)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = os.path.join(self.temp_dir.name, "conversations.json")
        self.repository = JsonConversationRepository(self.path)
        self.user_data, _ = self.repository.load_user("user-a")
        self.conversation, self.user_data = (
            self.repository.create_conversation("user-a")
        )

    class FakeStreamlit:
        class Container:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def __init__(self):
            self.session_state = {}
            self.markdown_calls = []
            self.error_calls = []
            self.code_calls = []

        def __getattr__(self, name):
            if name in {
                "title",
                "caption",
                "info",
                "warning",
                "markdown",
                "error",
                "code",
                "write",
                "subheader",
                "divider",
                "expander",
                "chat_message",
                "container",
                "popover",
            }:
                if name == "markdown":
                    return self.markdown_calls.append
                if name == "error":
                    return self.error_calls.append
                if name == "code":
                    return self.code_calls.append
                return lambda *args, **kwargs: self.Container()
            raise AttributeError(name)

        def button(self, *args, **kwargs):
            return False

        def columns(self, count):
            return [self for _ in range(count)]

        def chat_input(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

    def _process(self, agent, message="Question", submission_id=None, persist=True):
        return process_conversation_message(
            self.conversation,
            "user-a",
            message,
            lambda: agent,
            submission_id=submission_id,
            persist=(
                lambda updated: self.repository.save_conversation(
                    "user-a",
                    updated,
                )
                if persist
                else None
            ),
        )

    def test_conversation_is_the_default_dashboard_section(self):
        self.assertEqual(DASHBOARD_SECTIONS[0], "Conversation")
        self.assertEqual(DEFAULT_DASHBOARD_SECTION, "Conversation")

    def test_agent_interface_and_actual_markdown_response_are_persisted(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "A **real** answer."
        agent.last_response_metadata = {
            "memory_activity": [
                {"action": "retrieval", "status": "completed", "count": 2},
                {"action": "save", "status": "none"},
            ],
            "timings_ms": {
                "agent_execution": 120.0,
                "gemini_response": 100.0,
            },
            "polling": {"attempts": 2, "sleep_ms": 250.0},
            "memory_outcomes": [
                "memory_succeeded_without_memory_id",
            ],
            "memory_writes": 1,
        }

        result = self._process(
            agent,
            "  Help me plan.  ",
            submission_id="submission-1",
        )

        self.assertTrue(result["success"])
        agent.respond.assert_called_once_with("Help me plan.")
        self.assertEqual(
            [item["content"] for item in self.conversation["messages"]],
            ["Help me plan.", "A **real** answer."],
        )
        self.assertEqual(self.conversation["title"], "Help me plan.")
        self.assertEqual(
            self.conversation["messages"][1]["memory_activity"][0]["count"],
            2,
        )
        reloaded, malformed = JsonConversationRepository(self.path).load_user(
            "user-a"
        )
        self.assertFalse(malformed)
        assistant_message = reloaded["conversations"][0]["messages"][1]
        self.assertEqual(assistant_message["content"], "A **real** answer.")
        self.assertEqual(assistant_message["timings_ms"]["agent_execution"], 120)
        self.assertEqual(
            assistant_message["polling"],
            {"attempts": 2, "sleep_ms": 250.0},
        )
        self.assertEqual(
            assistant_message["memory_outcomes"],
            ["memory_succeeded_without_memory_id"],
        )
        self.assertEqual(assistant_message["memory_writes"], 1)

    def test_transcript_persists_across_repository_restart_and_stays_ordered(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.side_effect = ["First answer.", "Second answer."]
        self._process(agent, "First question")
        self._process(agent, "Second question")

        restarted_repository = JsonConversationRepository(self.path)
        user_data, malformed = restarted_repository.load_user("user-a")

        self.assertFalse(malformed)
        self.assertEqual(
            [
                message["content"]
                for message in user_data["conversations"][0]["messages"]
            ],
            [
                "First question",
                "First answer.",
                "Second question",
                "Second answer.",
            ],
        )

    def test_conversations_can_be_created_switched_renamed_and_deleted(self):
        second, user_data = self.repository.create_conversation("user-a")
        self.assertNotEqual(second["id"], self.conversation["id"])
        self.repository.set_active_conversation("user-a", self.conversation["id"])
        user_data, _ = self.repository.load_user("user-a")
        self.assertEqual(user_data["active_id"], self.conversation["id"])

        self.repository.rename_conversation(
            "user-a",
            second["id"],
            "Project planning",
        )
        self.repository.delete_conversation("user-a", second["id"])
        user_data, _ = self.repository.load_user("user-a")
        self.assertEqual(
            [item["id"] for item in user_data["conversations"]],
            [self.conversation["id"]],
        )
        self.assertEqual(user_data["active_id"], self.conversation["id"])

    def test_user_scoping_prevents_cross_user_transcript_access(self):
        other_conversation, _ = self.repository.create_conversation("user-b")
        agent = Mock()
        agent.user_id = "user-b"
        agent.respond.return_value = "Private answer."
        process_conversation_message(
            other_conversation,
            "user-b",
            "Private question",
            lambda: agent,
            persist=lambda updated: self.repository.save_conversation(
                "user-b",
                updated,
            ),
        )

        user_a, _ = self.repository.load_user("user-a")
        user_b, _ = self.repository.load_user("user-b")
        self.assertNotIn("Private question", repr(user_a))
        self.assertIn("Private question", repr(user_b))
        with self.assertRaises(ValueError):
            self.repository.save_conversation(
                "user-a",
                other_conversation,
            )

    def test_duplicate_submission_is_not_sent_twice(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "Done."

        first = self._process(agent, "Repeat-safe question", "same-event")
        duplicate = self._process(agent, "Repeat-safe question", "same-event")

        self.assertTrue(first["success"])
        self.assertTrue(duplicate["duplicate"])
        agent.respond.assert_called_once()
        self.assertEqual(len(self.conversation["messages"]), 2)

    def test_agent_failure_is_visible_and_does_not_record_a_success(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.side_effect = RuntimeError(
            "Gemini request failed: GEMINI_API_KEY=very-secret-value"
        )

        result = self._process(agent, "A question that failed")

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], GENERIC_AGENT_ERROR)
        self.assertEqual(len(self.conversation["messages"]), 1)
        failed_message = self.conversation["messages"][0]
        self.assertEqual(failed_message["status"], "failed")
        self.assertNotIn("very-secret-value", failed_message["error"])
        persisted, _ = self.repository.load_user("user-a")
        self.assertEqual(
            persisted["conversations"][0]["messages"][0]["status"],
            "failed",
        )

    def test_wrong_user_agent_is_rejected_before_calling_respond(self):
        agent = Mock()
        agent.user_id = "user-b"
        result = self._process(agent)

        self.assertFalse(result["success"])
        agent.respond.assert_not_called()
        self.assertEqual(self.conversation["messages"][0]["status"], "failed")

    def test_empty_response_is_failed(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = " "

        result = self._process(agent)

        self.assertFalse(result["success"])
        self.assertEqual(self.conversation["messages"][0]["status"], "failed")

    def test_storage_failure_before_agent_call_does_not_send_message(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "Not sent."
        with patch.object(
            self.repository,
            "save_conversation",
            side_effect=ConversationStoreError("disk unavailable"),
        ):
            result = self._process(agent)

        self.assertFalse(result["success"])
        agent.respond.assert_not_called()
        self.assertTrue(result.get("persistence_failed"))

    def test_first_message_survives_initial_store_failure_in_session_overlay(self):
        st = self.FakeStreamlit()
        user_data = {"active_id": None, "conversations": []}
        conversation = start_new_conversation(user_data, "user-a")
        result = process_conversation_message(
            conversation,
            "user-a",
            "Keep this prompt visible",
            lambda: self.fail("The agent must not run before the first save."),
            persist=Mock(side_effect=ConversationStoreError("disk unavailable")),
        )

        self.assertTrue(result["persistence_failed"])
        self.assertEqual(
            conversation["messages"][0]["content"],
            "Keep this prompt visible",
        )
        _stash_unsaved_conversation(st, "user-a", conversation)
        merged, error = _merge_unsaved_conversations(
            st,
            "user-a",
            {"active_id": None, "conversations": []},
            self.repository,
            readable=False,
        )

        self.assertIsNone(error)
        self.assertEqual(
            merged["conversations"][0]["messages"][0]["content"],
            "Keep this prompt visible",
        )

    def test_unsaved_conversation_overlay_is_retried_and_cleared(self):
        st = self.FakeStreamlit()
        conversation = self.conversation
        conversation["messages"].append(
            {
                "role": "user",
                "content": "Saved after storage recovered",
                "timestamp": "2026-10-25T00:00:00+00:00",
                "status": "failed",
                "error": "Storage was unavailable.",
            }
        )
        _stash_unsaved_conversation(st, "user-a", conversation)

        merged, error = _merge_unsaved_conversations(
            st,
            "user-a",
            {"active_id": None, "conversations": []},
            self.repository,
            readable=True,
        )

        self.assertIsNone(error)
        self.assertEqual(merged["active_id"], conversation["id"])
        self.assertNotIn(
            "user-a",
            st.session_state["dashboard_unsaved_conversations"],
        )
        loaded, malformed = self.repository.load_user("user-a")
        self.assertFalse(malformed)
        self.assertEqual(
            loaded["conversations"][0]["messages"][0]["content"],
            "Saved after storage recovered",
        )

    def test_storage_failure_after_agent_response_is_not_success_shaped(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "Generated but not durable."
        with patch.object(
            self.repository,
            "save_conversation",
            side_effect=[
                None,
                ConversationStoreError("disk unavailable"),
                ConversationStoreError("disk unavailable"),
            ],
        ):
            result = self._process(agent)

        self.assertFalse(result["success"])
        self.assertTrue(result["persistence_failed"])
        self.assertEqual(result["response"], "Generated but not durable.")
        self.assertEqual(len(self.conversation["messages"]), 1)
        self.assertEqual(self.conversation["messages"][0]["status"], "failed")

    def test_pending_request_is_recovered_as_failed_after_restart(self):
        self.conversation["messages"].append(
            {
                "role": "user",
                "content": "Interrupted question",
                "timestamp": "2026-10-25T00:00:00+00:00",
                "status": "pending",
            }
        )
        self.repository.save_conversation("user-a", self.conversation)

        reloaded, malformed = JsonConversationRepository(self.path).load_user(
            "user-a"
        )
        message = reloaded["conversations"][0]["messages"][0]
        self.assertTrue(malformed)
        self.assertEqual(message["status"], "failed")
        self.assertIn("interrupted", message["error"].casefold())

    def test_malformed_history_skips_invalid_entries_and_keeps_valid_transcripts(self):
        with open(self.path, "w", encoding="utf-8") as file:
            import json

            json.dump(
                {
                    "version": 1,
                    "users": {
                        "user-a": {
                            "active_id": "valid",
                            "conversations": [
                                {
                                    "id": "valid",
                                    "title": "Valid",
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Keep me",
                                            "timestamp": "today",
                                        }
                                    ],
                                },
                                "invalid conversation",
                            ],
                        }
                    },
                },
                file,
            )

        user_data, malformed = self.repository.load_user("user-a")
        self.assertTrue(malformed)
        self.assertEqual(len(user_data["conversations"]), 1)
        self.assertEqual(
            user_data["conversations"][0]["messages"][0]["content"],
            "Keep me",
        )

    def test_corrupt_store_is_not_silently_replaced(self):
        original = b"{corrupt"
        with open(self.path, "wb") as file:
            file.write(original)

        with self.assertRaises(ConversationStoreError):
            self.repository.load_user("user-a")
        with open(self.path, "rb") as file:
            self.assertEqual(file.read(), original)

    def test_new_conversation_never_touches_memory_or_graph(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "An answer."
        process_conversation_message(
            self.conversation,
            "user-a",
            "Keep this history",
            lambda: agent,
            persist=lambda updated: self.repository.save_conversation(
                "user-a",
                updated,
            ),
        )
        memory_delete = agent.memory.delete
        graph_save = agent.graph.save

        self.repository.create_conversation("user-a")

        memory_delete.assert_not_called()
        graph_save.assert_not_called()

    def test_conversation_rendering_does_not_create_an_agent(self):
        st = self.FakeStreamlit()
        empty_data, _ = self.repository.load_user("user-b")

        _render_conversation(st, "user-b", self.repository, empty_data)

        self.assertNotIn("dashboard_agents", st.session_state)

    def test_recent_search_matches_titles_and_all_messages(self):
        self.conversation["title"] = "Weekly review"
        self.conversation["messages"] = [
            {
                "role": "user",
                "content": "Plan project phoenix",
                "timestamp": "2026-10-25T00:00:00+00:00",
                "status": "success",
            },
            {
                "role": "assistant",
                "content": "The response mentions aurora.",
                "timestamp": "2026-10-25T00:01:00+00:00",
            },
        ]
        self.assertEqual(
            len(_recent_conversations([self.conversation], "phoenix")),
            1,
        )
        self.assertEqual(
            len(_recent_conversations([self.conversation], "aurora")),
            1,
        )
        self.assertEqual(
            len(_recent_conversations([self.conversation], "weekly")),
            1,
        )
        self.assertEqual(
            len(_recent_conversations([self.conversation], "missing")),
            0,
        )

    def test_message_role_uses_native_markdown_renderer(self):
        st = self.FakeStreamlit()
        _render_conversation_message(
            st,
            {
                "role": "assistant",
                "content": "**Formatted answer**",
                "timestamp": "2026-10-25T00:00:00+00:00",
                "latency_ms": 10,
                "memory_activity": [],
            },
            "assistant-message",
        )
        self.assertEqual(st.markdown_calls, ["**Formatted answer**"])

    def test_command_responses_are_visually_separate_from_agent_answers(self):
        st = self.FakeStreamlit()
        _render_conversation_message(
            st,
            {
                "role": "assistant",
                "content": "Could not retrieve memories: unavailable",
                "timestamp": "2026-10-25T00:00:00+00:00",
                "message_type": "command",
                "command_name": "memories",
                "command_status": "error",
                "command_format": "text",
            },
            "command-message",
        )

        self.assertEqual(
            st.error_calls,
            ["Could not retrieve memories: unavailable"],
        )
        self.assertNotIn(
            "Could not retrieve memories: unavailable",
            st.markdown_calls,
        )

    def test_normalizer_rejects_conversation_owned_by_another_user(self):
        normalized, malformed = normalize_user_conversations(
            {
                "active_id": "foreign",
                "conversations": [
                    {
                        "id": "foreign",
                        "owner_id": "user-b",
                        "title": "Private",
                        "created_at": "2026-10-25T00:00:00+00:00",
                        "messages": [],
                    }
                ],
            },
            "user-a",
        )

        self.assertTrue(malformed)
        self.assertEqual(normalized["conversations"], [])
        self.assertIsNone(normalized["active_id"])


if __name__ == "__main__":
    unittest.main()
