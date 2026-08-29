import os
import logging
import json

logging.getLogger("google").setLevel(logging.ERROR)

from google import genai
from google.genai import types

from app.memory import MemoryManager
from app.config import USER_ID


class Agent:
    """
    AI agent with long-term memory.
    """

    def __init__(self):
        self.memory = MemoryManager()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please add your api key in the environment."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.conversation_history = []
        self.max_history = 10

    def remember(self, message, response):
        """
        Store the conversation in Mem0.
        """

        messages = [
            {
                "role": "user",
                "content": message
            }
        ]

        try:
            return self.memory.add(
                messages,
                USER_ID
            )

        except Exception as e:
            print(f"\n[ERROR] Memory storage failed: {e}\n")
            return None

    def recall(self, query):
        """
        Search the user's long-term memories.
        """

        try:
            return self.memory.search(
                query,
                USER_ID
            )

        except Exception as e:
            print(f"\n[ERROR] Memory search failed: {e}\n")
            return {"results": []}
    
    def check_for_update(self, message):
        """
        Check wether the new message may update an existing memory.
        """

        update_phrases = [
            "no longer",
            "don't like",
            "do not like",
            "doesn't like",
            "does not like",
            "instead",
            "now",
            "anymore",
            "changed",
        ]

        message_lower = message.lower()

        return any(
            phrase in message_lower
            for phrase in update_phrases
        )

    def update_memory(self, message):
        """
        Find and replace a specific outdated memory.
        """

        try:
            memories = self.memory.get_all(USER_ID)

            if not memories or "results" not in memories:
                return

            message_lower = message.lower()

            # Check whether this is a negative preference update
            if "don't like" in message_lower:
                changed_item = message_lower.split("don't like", 1)[1]

            elif "do not like" in message_lower:
                changed_item = message_lower.split("do not like", 1)[1]

            else:
                return

            # Clean the changed item
            changed_item = changed_item.replace("anymore", "")
            changed_item = changed_item.strip(" .,!?:;'\"")

            # Create simple singular/plural variations
            item_variations = {
                changed_item,
                changed_item.rstrip("s"),
            }

            for memory in memories["results"]:
                memory_text = memory["memory"].lower()

                # Check whether this memory contains the changed item
                if any(
                    variation and variation in memory_text
                    for variation in item_variations
                ):
                    print(f"[MEMORY] Updating: {memory['memory']}")
                    self.memory.delete(memory["id"])

        except Exception as e:
            print(f"\n[ERROR] Memory update failed: {e}\n")

    def decide_memory(self, message):
        """
        Decide whether the user's message contains information
        that should be stored as long-term memory.
        """

        prompt = f"""
You are a memory decision system.

Decide whether the user's message contains information
that would be useful to remember for future conversations.

Remember things such as:
- personal information
- preferences
- interests
- goals
- projects
- skills
- important relationships
- recurring activities
- important facts about the user

Do not remember:
- greetings
- casual conversation
- temporary questions
- requests for information
- ordinary assistant interactions
- information that is clearly irrelevant in future conversations

The user's message is:

"{message}"

Return ONLY valid JSON in this exact format:

{{
    "should_remember": true
}}

or

{{
    "should_remember": false
}}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )
            

            result = json.loads(response.text)

            return result.get("should_remember", False)

        except Exception as e:
            print(f"\n[ERROR] Gemini request failed: {e}\n")
            return False

    def respond(self, message):
        """
        Generate a response using short-term
        and long-term memory.
        """

        # Get relevant long-term memories
        memories = self.recall(message)

        memory_context = ""

        if memories and "results" in memories:

            relevant_memories = [
                item
                for item in memories["results"]
                if item.get("score", 0) >= 0.15
            ]

            #Keep only the top 5 most relevant memories.
            relevant_memories = sorted(
                relevant_memories,
                key=lambda item: item.get("score", 0),
                reverse=True
            )[:5]

            memory_context = "\n".join(
                item["memory"]
                for item in relevant_memories
            )

        # Add user's message to short-term memory
        self.conversation_history.append({
            "role": "user",
            "text": message
        })

        # Keep only recent conversation
        recent_history = self.conversation_history[-self.max_history:]

        conversation_context = "\n".join(
            f"{item['role']}: {item['text']}"
            for item in recent_history
        )

        # Give Gemini both types of context
        system_prompt = f"""
You are a helpful AI assistant with long-term memory.

Relevant long-term memories:
{memory_context}

Recent conversation:
{conversation_context}

Current user message:
{message}

Use both types of context when relevant.

The current user message is the most important input.
Use recent conversation to understand context and references.

When recent conversation contains newer information that conflicts
with older information, always prefer the most recent information.

When answering a question about the current state of something,
use the most recent information as the current truth.

Do not bring up older states unless the user asks about them.
Do not add unnecessary commentary about changes from earlier.

Use long-term memories silently.
Never mention memories or how you know something.
Never say "based on what you told me", "you told me",
"according to my memory", or similar phrases.

Do not invent information.
Answer naturally.
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )

        answer = response.text

        # Add AI response to short-term memory
        self.conversation_history.append({
            "role": "assistant",
            "text": answer
        })

        # Store only potentially useful information
        if self.check_for_update(message):

            self.update_memory(message)

        else:

            if self.decide_memory(message):
                self.remember(message, answer)

        return answer