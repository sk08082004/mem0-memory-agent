from mem0 import Memory


class MemoryManager:
    """
    Handles long-term memory using Mem0.
    """

    def __init__(self):
        self.memory = None
        self.initialized = False

    def initialize(self):
        """
        Initialize Mem0 with Ollama.
        """

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "qwen3:4b",
                    "ollama_base_url": "http://localhost:11434",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                }
            },

            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest",
                    "ollama_base_url": "http://localhost:11434",
                }
            },

            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": "./data/qdrant"
                }
            },

            "version": "v1.1",
        }

        self.memory = Memory.from_config(config)

        self.initialized = True

        print("Mem0 memory system ready.")

    def add(self, messages, user_id):
        """
        Extract and store important information.
        """

        if not self.initialized:
            raise RuntimeError("Memory system is not initialized.")

        result = self.memory.add(
            messages,
            user_id=user_id
        )

        print("[MEMORY] Added/updated memory.")

        return result

    def search(self, query, user_id):
        """
        Search long-term memory.
        """

        if not self.initialized:
            raise RuntimeError("Memory system is not initialized.")

        results = self.memory.search(
            query=query,
            user_id=user_id
        )

        return results

    def get_all(self, user_id):
        """
        Retrieve all memories for a user.
        """

        if not self.initialized:
            raise RuntimeError("Memory system is not initialized.")

        return self.memory.get_all(
            user_id=user_id
        )

    def delete(self, memory_id):
        """
        Delete a specific memory.
        """

        if not self.initialized:
            raise RuntimeError("Memory system is not initialized.")

        return self.memory.delete(memory_id)