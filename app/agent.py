import os

from google import genai

from app.memory import MemoryManager
from app.config import USER_ID


class Agent:
    """
    AI agent with long-term memory.
    """

    def __init__(self):
        self.memory = MemoryManager()

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def remember(self, message, response):
        """
        Store the conversation in Mem0.
        """

        messages = [
            {
                "role": "user",
                "content": message
            },
            {
                "role": "assistant",
                "content": response
            }
        ]

        return self.memory.add(
            messages,
            USER_ID
        )

    def recall(self, query):
        """
        Search the user's long-term memories.
        """

        return self.memory.search(
            query,
            USER_ID
        )

    def respond(self, message):
        """
        Generate a response using Gemini
        and relevant long-term memories.
        """

        # 1. Search memory
        memories = self.recall(message)

        print("\nRelevant memories:")
        print(memories)

        # 2. Prepare memory context
        memory_context = ""

        if memories and "results" in memories:
         memory_context = "\n".join(
         item["memory"]
         for item in memories["results"]
        )

        # 3. Build prompt
        system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Here are relevant memories about the user:

{memory_context}

Use these memories when they are relevant.
Do not invent memories.
If there are no relevant memories, simply answer normally.
"""

        # 4. Ask Gemini
        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": message
                        }
                    ]
                }
            ]
        )

        answer = response.text

        # 5. Store conversation
        self.remember(
            message,
            answer
        )

        return answer