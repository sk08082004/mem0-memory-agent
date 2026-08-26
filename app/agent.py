from ollama import chat

from app.memory import MemoryManager
from app.config import USER_ID


class Agent:
    """
    AI agent with long-term memory.
    """

    def __init__(self):
        self.memory = MemoryManager()
        self.memory.initialize()

    def remember(self, message):
        """
        Store important information from the conversation.
        """

        messages = [
            {
                "role": "user",
                "content": message
            }
        ]

        return self.memory.add(
            messages,
            USER_ID
        )

    def recall(self, query):
        """
        Search long-term memory.
        """

        return self.memory.search(
            query,
            USER_ID
        )

    def respond(self, message):
        """
        Generate an AI response using Qwen3.
        """

        # 1. Search long-term memory
        memories = self.recall(message)

        print("\nRelevant memories:")
        print(memories)

        # 2. Convert memories into context
        memory_context = ""

        if memories:
            memory_context = "\n".join(
                str(memory)
                for memory in memories
            )

        # 3. Build the prompt
        system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Relevant information remembered about the user:

{memory_context}

Use the remembered information when it is relevant.
Do not invent memories.
"""

        # 4. Ask Qwen3
        response = chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        answer = response["message"]["content"]

        # 5. Store the conversation in Mem0
        self.remember(message)

        return answer