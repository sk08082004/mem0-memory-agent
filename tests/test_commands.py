import unittest
from unittest.mock import Mock, patch

from app.commands import (
    COMMANDS,
    command_requires_agent,
    dispatch_chat_command,
    format_command_help,
)
from app.conversation import (
    normalize_user_conversations,
    process_conversation_command,
    process_conversation_message,
)
from app.dashboard import _conversation_agent


class CommandTests(unittest.TestCase):
    def test_help_is_generated_from_command_registry_and_lists_all_commands(self):
        help_text = format_command_help()

        for syntax, description in COMMANDS:
            with self.subTest(syntax=syntax):
                self.assertIn(syntax, help_text)
                self.assertIn(description, help_text)
        self.assertIn("/related <memory_number>", help_text)
        self.assertIn("--restore-local-source <memory_id>", help_text)

    def test_memories_uses_the_same_active_memory_filter_as_cli(self):
        agent = Mock()
        agent.user_id = "user-a"
        current = [{"id": "m1", "memory": "Prefers Python"}]
        with patch("app.main.get_active_memories", return_value=current) as get_active:
            result = dispatch_chat_command(agent, "/memories")

        get_active.assert_called_once_with(agent)
        self.assertEqual(result["status"], "success")
        self.assertIn("Prefers Python", result["text"])
        self.assertEqual(result["command"], "memories")

    def test_command_errors_and_unknown_commands_do_not_call_agent_response(self):
        agent = Mock()
        agent.user_id = "user-a"

        invalid = dispatch_chat_command(agent, "/related not-a-number")
        unknown = dispatch_chat_command(agent, "/not-a-command")

        self.assertEqual(invalid["status"], "error")
        self.assertIn("valid memory number", invalid["text"])
        self.assertEqual(unknown["status"], "error")
        agent.respond.assert_not_called()

    def test_clear_requires_confirmation_before_mutating_memories(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.memory.get_all_complete.return_value = {
            "count": 2,
            "results": [{}, {}],
        }

        pending = dispatch_chat_command(agent, "/clear")

        self.assertEqual(pending["status"], "confirmation")
        self.assertEqual(pending["action"]["type"], "confirm_clear")
        agent.memory.clear.assert_not_called()

    def test_clear_confirmation_calls_existing_clear_and_graph_handlers(self):
        agent = Mock()
        agent.user_id = "user-a"
        agent.memory.get_all_complete.return_value = {
            "count": 2,
            "results": [{}, {}],
        }
        agent.memory.clear.return_value = {
            "retrieval_complete": True,
            "remaining_retrieval_complete": True,
            "remaining_memories": {"count": 0, "results": []},
            "complete": True,
            "deleted_ids": ["m1", "m2"],
            "failed_ids": [],
            "unconfirmed_ids": [],
            "remaining_ids": [],
        }

        result = dispatch_chat_command(agent, "/clear", confirm_clear=True)

        agent.memory.clear.assert_called_once_with("user-a")
        agent.graph.sync_user_graph.assert_called_once_with(
            "user-a",
            {"count": 0, "results": []},
        )
        self.assertEqual(result["status"], "success")

    def test_supported_command_shapes_and_exit_do_not_invoke_gemini(self):
        agent = Mock()
        agent.user_id = "user-a"
        for text in (
            "/help",
            "/user",
            "/new",
            "/login user-b",
            "/logout",
            "exit",
        ):
            with self.subTest(text=text):
                self.assertTrue(dispatch_chat_command(agent, text)["handled"])
        agent.respond.assert_not_called()

    def test_relationship_recovery_command_uses_existing_agent_method(self):
        agent = Mock()
        agent.user_id = "user-a"
        report = {"dry_run": False, "attempted": 1}
        agent.recover_relationships.return_value = report

        result = dispatch_chat_command(
            agent,
            "/recover-relationships --apply",
        )

        agent.recover_relationships.assert_called_once_with(
            dry_run=False,
            source_id=None,
        )
        self.assertEqual(result["format"], "code")
        self.assertIn('"attempted": 1', result["text"])

    def test_mem0_event_reconciliation_command_uses_agent_reconciliation(self):
        agent = Mock()
        agent.reconcile_pending_memory_events.return_value = {
            "checked": 1,
            "synchronized": 0,
            "pending": 1,
            "failed": 0,
            "details": [
                {
                    "operation_type": "add",
                    "status": "outcome_uncertain",
                }
            ],
        }

        result = dispatch_chat_command(
            agent,
            "/reconcile-memory-events 3",
        )

        agent.reconcile_pending_memory_events.assert_called_once_with(limit=3)
        self.assertEqual(result["status"], "success")
        self.assertIn("Still pending: 1", result["text"])
        self.assertIn("outcome_uncertain", result["text"])
        self.assertTrue(command_requires_agent("/reconcile-memory-events"))

    def test_mem0_event_reconciliation_rejects_invalid_limits(self):
        agent = Mock()
        for text in (
            "/reconcile-memory-events 0",
            "/reconcile-memory-events invalid",
            "/reconcile-memory-events 1 extra",
        ):
            with self.subTest(text=text):
                result = dispatch_chat_command(agent, text)
                self.assertEqual(result["status"], "error")
        agent.reconcile_pending_memory_events.assert_not_called()

    def test_command_messages_are_not_reloaded_as_gemini_chat_context(self):
        conversation = {
            "id": "conversation-1",
            "messages": [
                {
                    "role": "user",
                    "content": "/help",
                    "status": "success",
                    "message_type": "command",
                },
                {
                    "role": "assistant",
                    "content": "Available commands",
                    "message_type": "command",
                },
                {
                    "role": "user",
                    "content": "Remember my preference for Python.",
                    "status": "success",
                },
                {
                    "role": "assistant",
                    "content": "I will remember that.",
                },
            ],
        }
        agent = Mock()
        agent.user_id = "user-a"
        agent.max_history = 10
        st = Mock()
        st.session_state = {}

        with patch("app.main.create_agent", return_value=agent):
            loaded = _conversation_agent(st, "user-a", conversation)

        self.assertIs(loaded, agent)
        self.assertEqual(
            agent.conversation_history,
            [
                {
                    "role": "user",
                    "text": "Remember my preference for Python.",
                },
                {"role": "assistant", "text": "I will remember that."},
            ],
        )

    def test_natural_language_is_not_classified_as_a_command(self):
        prompt = "What do you remember about my programming preferences?"

        self.assertIsNone(dispatch_chat_command(None, prompt))
        self.assertFalse(command_requires_agent(prompt))

        conversation = {
            "id": "conversation-1",
            "owner_id": "user-a",
            "title": "New conversation",
            "created_at": "2026-10-25T00:00:00+00:00",
            "messages": [],
        }
        agent = Mock()
        agent.user_id = "user-a"
        agent.respond.return_value = "You prefer Python."
        result = process_conversation_message(
            conversation,
            "user-a",
            prompt,
            lambda: agent,
        )

        self.assertTrue(result["success"])
        agent.respond.assert_called_once_with(prompt)

    def test_command_responses_persist_as_distinct_messages_and_deduplicate(self):
        conversation = {
            "id": "conversation-1",
            "owner_id": "user-a",
            "title": "New conversation",
            "created_at": "2026-10-25T00:00:00+00:00",
            "messages": [],
        }
        result = dispatch_chat_command(None, "/help")
        persist = Mock()

        first = process_conversation_command(
            conversation,
            "user-a",
            "/help",
            result,
            submission_id="event-1",
            persist=persist,
        )
        duplicate = process_conversation_command(
            conversation,
            "user-a",
            "/help",
            result,
            submission_id="event-1",
            persist=persist,
        )

        self.assertTrue(first["success"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(len(conversation["messages"]), 2)
        self.assertEqual(conversation["messages"][0]["message_type"], "command")
        self.assertEqual(
            conversation["messages"][1]["message_type"],
            "command",
        )
        self.assertEqual(persist.call_count, 1)
        normalized, malformed = normalize_user_conversations(
            {"active_id": conversation["id"], "conversations": [conversation]},
            "user-a",
        )
        self.assertFalse(malformed)
        stored_response = normalized["conversations"][0]["messages"][1]
        self.assertEqual(stored_response["command_format"], "markdown")
        self.assertEqual(stored_response["command_status"], "success")

    def test_help_matches_command_list_shown_to_cli(self):
        from app.main import format_command_help as cli_help

        self.assertEqual(cli_help(), format_command_help())


if __name__ == "__main__":
    unittest.main()
