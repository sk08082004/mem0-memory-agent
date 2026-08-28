import os

from mem0 import MemoryClient


class MemoryManager:
    """
    Handles long-term memory using Mem0 Cloud.
    """

    def __init__(self):
        api_key = os.getenv("MEM0_API_KEY")

        if not api_key:
            raise ValueError("MEM0_API_KEY is not set.")

        self.client = MemoryClient(api_key=api_key)

    def add(self, messages, user_id):
        """
        Store important information from a conversation.
        """

        result = self.client.add(
            messages=messages,
            user_id=user_id
        )

        
        return result

    def search(self, query, user_id):
        """
        Search the user's long-term memories.
        """

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

        return self.client.get_all(
            filters={
                "user_id": user_id
            }
        )

    def delete(self, memory_id):
        """
        Delete one memory.
        """

        return self.client.delete(memory_id)


    def clear(self, user_id):
      """
      Delete all memories belonging to a user.
      """

      memories = self.client.get_all(
        filters={
            "user_id": user_id
           }
        )

      for memory in memories["results"]:
        self.client.delete(memory["id"])

      return True