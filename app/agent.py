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
        Store information from the user's message.
        """

        messages = [
            {
                "role": "user",
                "content": message
            }
        ]

        self.memory.add(
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
        Generate a response.

        The actual AI model will be connected later.
        """

        memories = self.recall(message)

        print("\nRelevant memories:")
        print(memories)

        return (
            "[MODEL NOT CONNECTED YET]\n"
            f"You said: {message}"
        )