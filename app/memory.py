import os

from mem0 import MemoryClient
from app.performance import timed_operation


class MemoryManager:
    """
    Handles long-term memory using Mem0 Cloud.
    """

    def __init__(self):
        api_key = os.getenv("MEM0_API_KEY")

        if not api_key:
            raise ValueError("MEM0_API_KEY is not set.")

        self.client = MemoryClient(api_key=api_key)

    def add(self, messages, user_id, metadata=None):
        """
        Store important information from a conversation.
        """

        with timed_operation("Mem0 add"):
            result = self.client.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata
            )

        return result

    def get_event(self, event_id):
        """
        Get the status of a Mem0 background event.
        """

        return self.client.client.get(
            f"/v1/event/{event_id}/"
        )

    def search(self, query, user_id):
        """
        Search the user's long-term memories.
        """

        with timed_operation("Mem0 search/recall"):
            results = self.client.search(
                query=query,
                filters={
                    "user_id": user_id
                }
            )

        return results

    def get_all(self, user_id):
        """
        Get all memories belonging to the user.
        """

        with timed_operation("Mem0 list"):
            results = self.client.get_all(
                filters={
                    "user_id": user_id
                }
            )

        return results

    def get_all_complete(self, user_id):
        """
        Retrieve every page of memories for one user.
        """

        with timed_operation("Mem0 complete pagination"):
            return self._get_all_complete(user_id)

    def _get_all_complete(self, user_id):
        memories = []
        page = 1
        expected_count = None

        while True:
            with timed_operation(f"Mem0 list page {page}"):
                response = self.client.get_all(
                    filters={
                        "user_id": user_id
                    },
                    page=page
                )

            if (
                not isinstance(response, dict)
                or not isinstance(response.get("results"), list)
                or type(response.get("count")) is not int
                or response["count"] < 0
                or "next" not in response
                or (
                    response["next"] is not None
                    and (
                        not isinstance(response["next"], str)
                        or not response["next"].strip()
                    )
                )
            ):
                raise ValueError(
                    "Mem0 returned an incomplete or invalid memory listing."
                )

            if expected_count is None:
                expected_count = response["count"]
            elif response["count"] != expected_count:
                raise ValueError(
                    "Mem0 memory count changed during paginated retrieval."
                )

            page_memories = response["results"]
            if any(
                not isinstance(memory, dict)
                or not isinstance(memory.get("id"), str)
                or not memory["id"].strip()
                or not isinstance(memory.get("memory"), str)
                or not memory["memory"].strip()
                for memory in page_memories
            ):
                raise ValueError(
                    "Mem0 returned a memory without a valid ID or usable text."
                )

            memories.extend(page_memories)

            if response["next"] is None:
                break

            if not page_memories or len(memories) >= expected_count:
                raise ValueError(
                    "Mem0 pagination did not match the reported memory count."
                )

            page += 1

        memory_ids = [memory["id"] for memory in memories]
        if (
            len(memories) != expected_count
            or len(memory_ids) != len(set(memory_ids))
        ):
            raise ValueError(
                "Mem0 pagination did not return a complete unique memory list."
            )

        return {
            "count": expected_count,
            "results": memories,
        }

    def get_by_id(self, memory_id):
        """
        Get one specific memory by its ID.
        """

        return self.client.get(memory_id)

    def delete(self, memory_id):
        """
        Delete one memory.
        """

        with timed_operation("Mem0 delete"):
            return self.client.delete(memory_id)

    @staticmethod
    def delete_result_status(result):
        if not isinstance(result, dict) or not result:
            return "unconfirmed"

        positive_indicators = []
        negative_indicators = []
        malformed_indicators = []

        for key in ("success", "deleted"):
            if key not in result:
                continue
            if result[key] is True:
                positive_indicators.append(key)
            elif result[key] is False:
                negative_indicators.append(key)
            else:
                malformed_indicators.append(key)

        status = result.get("status")
        if isinstance(status, str):
            normalized_status = status.strip().lower()
            if normalized_status in {"success", "succeeded", "deleted"}:
                positive_indicators.append("status")
            elif normalized_status in {"failed", "failure", "error"}:
                negative_indicators.append("status")
            else:
                malformed_indicators.append("status")
        elif "status" in result:
            malformed_indicators.append("status")

        message = result.get("message")
        if isinstance(message, str):
            normalized_message = " ".join(
                message.lower().replace("!", "").replace(".", "").split()
            )
            if normalized_message in {
                "deleted successfully",
                "memory deleted successfully",
                "memory was deleted successfully",
            }:
                positive_indicators.append("message")
            elif normalized_message in {
                "delete failed",
                "memory deletion failed",
                "failed to delete memory",
            }:
                negative_indicators.append("message")
            else:
                malformed_indicators.append("message")
        elif "message" in result:
            malformed_indicators.append("message")

        if positive_indicators and negative_indicators:
            return "unconfirmed"
        if negative_indicators:
            return "failed"
        if malformed_indicators or not positive_indicators:
            return "unconfirmed"

        return "confirmed"

    def clear(self, user_id):
        """
        Delete all memories for one user and report each outcome.
        """

        try:
            memories = self.get_all_complete(user_id)
        except Exception as e:
            return {
                "retrieval_complete": False,
                "deleted_ids": [],
                "failed_ids": [],
                "unconfirmed_ids": [],
                "remaining_ids": [],
                "remaining_memories": None,
                "remaining_retrieval_complete": False,
                "error": str(e),
            }

        deleted_ids = []
        failed_ids = []
        unconfirmed_ids = []

        for memory in memories["results"]:
            memory_id = memory["id"]
            try:
                status = self.delete_result_status(self.delete(memory_id))
            except Exception:
                status = "unconfirmed"

            if status == "confirmed":
                deleted_ids.append(memory_id)
            elif status == "failed":
                failed_ids.append(memory_id)
            else:
                unconfirmed_ids.append(memory_id)

        try:
            remaining = self.get_all_complete(user_id)
        except Exception as e:
            return {
                "retrieval_complete": True,
                "deleted_ids": deleted_ids,
                "failed_ids": failed_ids,
                "unconfirmed_ids": unconfirmed_ids,
                "remaining_ids": sorted(
                    set(failed_ids + unconfirmed_ids)
                ),
                "remaining_memories": None,
                "remaining_retrieval_complete": False,
                "error": str(e),
            }

        remaining_ids = {
            memory["id"]
            for memory in remaining["results"]
        }
        confirmed_ids = set(deleted_ids)
        confirmed_ids.update(
            memory_id
            for memory_id in failed_ids + unconfirmed_ids
            if memory_id not in remaining_ids
        )

        return {
            "retrieval_complete": True,
            "deleted_ids": sorted(confirmed_ids),
            "failed_ids": [
                memory_id for memory_id in failed_ids
                if memory_id in remaining_ids
            ],
            "unconfirmed_ids": [
                memory_id for memory_id in unconfirmed_ids
                if memory_id in remaining_ids
            ],
            "remaining_ids": sorted(remaining_ids),
            "remaining_memories": remaining,
            "remaining_retrieval_complete": True,
            "complete": not remaining_ids,
            "error": None,
        }