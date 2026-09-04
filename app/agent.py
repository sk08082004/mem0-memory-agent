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
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30000
            )
        )

        self.conversation_history = []
        self.max_history = 10

    def remember(self, message, response):
        """
        Extract important long-term information from the user's message
        and assign an importance score to each memory.
        """

        prompt = f"""
You are a long-term memory extraction system for an AI agent.

Your job is to extract information from the user's message that
could be useful in future conversations.

Do not use a fixed list of categories.

Judge each piece of information based on its long-term usefulness.

Remember information such as:
- Important personal context
- Preferences
- Relationships
- Projects
- Technical decisions
- Ongoing work
- Plans
- Goals
- Future events
- Important experiences
- Substantial context that may help the agent later

Do NOT remember information that is only temporary or useful
for the immediate moment, unless it has additional long-term value.

Do not invent information.
Only extract information explicitly supported by the user's message.

For every memory, assign an importance score from 1 to 10.

Importance scale:

1-3:
Minor information with little future usefulness.

4-6:
Moderately useful information that may help in future conversations.

7-8:
Important information that provides meaningful long-term context.

9-10:
Highly important information that would substantially improve
future conversations or represents very important long-term context.

Prefer remembering useful information over aggressively filtering it.

Return ONLY valid JSON in this exact format:

{{
    "memories": [
        {{
            "text": "standalone factual memory",
            "importance": 8
        }}
    ]
}}

If there is nothing worth remembering, return:

{{
    "memories": []
}}

User message:

{message}
"""

        try:
            extraction = self.client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                )
            )

            response_text = extraction.text.strip()

            if not response_text:
                print("[MEMORY] Gemini returned an empty response.")
                return None

            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            result = json.loads(response_text)

            memories = result.get("memories", [])

            if not memories:
                return None

            for memory in memories:

                memory_text = memory.get("text", "").strip()
                importance = memory.get("importance", 5)

                if not memory_text:
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": memory_text
                    }
                ]

                self.memory.add(
                    messages,
                    USER_ID,
                    metadata={
                        "importance": importance
                    }
                )

            return True

        except Exception as e:
            print(
                f"\n[ERROR] Memory extraction/storage failed: {e}"
            )

            if "extraction" in locals():
                print("[DEBUG] Gemini memory response:")
                print(extraction.text)

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
            print(
                f"\n[ERROR] Memory search failed: {e}\n"
            )
            return {"results": []}

    def check_for_update(self, message):
        """
        Check whether the new message may update an existing memory.
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
        Use Gemini to identify outdated memories that conflict
        with the user's newest message.
        """

        try:
            # Get all existing memories for the user.
            memories = self.memory.get_all(USER_ID)

            if not memories or "results" not in memories:
                return

            existing_memories = memories["results"]

            if not existing_memories:
                return

            # Give Gemini the existing memories and the new message.
            memory_list = "\n".join(
                f"ID: {memory['id']} | Memory: {memory['memory']}"
                for memory in existing_memories
            )

            prompt = f"""
You are a memory conflict detection system.

The user has provided a new message.

Your job is to determine whether the new message
contradicts or updates any existing long-term memories.

Existing memories:

{memory_list}

New user message:

"{message}"

Rules:

1. Only identify a memory as outdated if the new message
clearly changes, contradicts, or replaces it.

2. Do not assume information that the user did not provide.

3. Do not delete memories merely because they are related
to the new message.

4. If the new message does not conflict with an existing
memory, do not mark it for deletion.

5. A newer preference or decision should replace an older
preference or decision when they clearly conflict.

6. Return only the IDs of memories that are definitely outdated.

Return ONLY valid JSON in this format:

{{
    "outdated_memory_ids": [
        "memory-id-1",
        "memory-id-2"
    ]
}}

If there are no conflicting memories, return:

{{
    "outdated_memory_ids": []
}}
"""

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

            outdated_ids = result.get(
                "outdated_memory_ids",
                []
            )

            # Delete only memories that Gemini identified
            # as definitely outdated.
            for memory_id in outdated_ids:

                valid_memory = any(
                    memory["id"] == memory_id
                    for memory in existing_memories
                )

                if valid_memory:
                    print(
                        f"[MEMORY] Removing outdated memory: {memory_id}"
                    )
                    self.memory.delete(memory_id)

        except Exception as e:
            print(
                f"\n[ERROR] Memory conflict resolution failed: {e}\n"
            )

    def decide_memory(self, message):
        """
        Decide whether the user's message contains information
        that should be stored as long-term memory.
        """

        prompt = f"""
You are the long-term memory decision system for an AI agent.

Your job is to decide whether the user's message contains information
that should be stored in long-term memory for future conversations.

Do NOT use a fixed list of categories or keywords to make this decision.
The user may provide any kind of information, and potentially important
information can appear in completely unexpected forms.

Instead, judge the information based on its long-term value and
potential usefulness in future conversations.

Store information when it is likely to remain useful beyond the current
conversation or situation. This includes information that helps the
agent understand the user, their history, their projects, their work,
their interests, their relationships, their decisions, their preferences,
their plans, their experiences, their knowledge, or important context
they have shared.

Also remember substantial context from conversations when that context
could help the agent understand or continue the user's work in a future
conversation. If the user explains a project, system, idea, workflow,
research, plan, or other ongoing subject in meaningful detail, preserve
the important information from it rather than remembering only isolated
sentences.

The information does NOT need to be permanent to be useful. Information
can still be worth remembering if it is likely to remain relevant for a
reasonable period of time or could help the agent provide better context
in a future conversation.

Distinguish between a momentary state and information that has future
relevance.

Do NOT store information merely because it is happening right now when
it has no meaningful relevance beyond the current moment.

For example:

"Sandeep is eating right now."
→ temporary state → normally do not remember.

"Sandeep is sleeping right now."
→ temporary state → normally do not remember.

"It is raining outside right now."
→ temporary observation → normally do not remember.

"Someone is watching TV right now."
→ temporary state → normally do not remember.

However, remember information that describes a future event, plan,
deadline, intention, expectation, or upcoming activity, even if that
information is time-sensitive.

For example:

"ISRO is going to do a launch next month."
→ future event → remember.

"My exam is next month."
→ future deadline → remember.

"I am planning to visit Delhi next month."
→ future plan → remember.

"My project presentation is on Friday."
→ upcoming event → remember.

The fact that information will eventually become outdated does NOT
automatically make it unworthy of memory. Consider whether it could be
useful in a future conversation before deciding.

The same principle applies to information that is not explicitly about
the user. Important information about projects, organizations, people,
events, research, or other subjects can be worth remembering when it
provides meaningful context for future conversations.

For example:

"ISRO is working on a new mission."
→ potentially useful subject context → consider remembering.

"The launch is scheduled for next month."
→ future event → remember.

"ISRO launched a rocket today."
→ current event → normally do not remember unless the context makes
it particularly important or useful later.

The important distinction is:

Momentary and contextless → normally do not remember.
Meaningful or future-relevant → remember.
Potentially useful context → consider remembering.

When a message contains a mixture of temporary and meaningful
information, remember the meaningful information and ignore the
temporary details.

Prefer remembering useful information over aggressively filtering it.
Do not be overly restrictive.

When uncertain, ask yourself:

"If the user talks to this agent again days, weeks, or months from now,
could knowing this information make the agent substantially more useful?"

If yes, remember it.

If the information is only useful for the immediate moment and is
unlikely to matter later, do not remember it.

Do not invent information or infer facts that the user did not actually
provide.

Return ONLY valid JSON in this exact format:

{{
    "should_remember": true
}}

or

{{
    "should_remember": false
}}

User message:

"{message}"
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

            return result.get(
                "should_remember",
                False
            )

        except Exception as e:
            print(
                f"\n[ERROR] Gemini memory decision failed: {e}\n"
            )
            return False

    def respond(self, message):
        """
        Generate a response using short-term
        and long-term memory.
        """

        # Get relevant long-term memories.
        memories = self.recall(message)

        memory_context = ""

        if memories and "results" in memories:

            relevant_memories = [
                item
                for item in memories["results"]
                if item.get("score", 0) >= 0.15
            ]

            # Combine semantic relevance, importance, and recency.
            for item in relevant_memories:

                # Semantic relevance from Mem0.
                relevance_score = float(
                    item.get("score", 0)
                )

                # Importance assigned when the memory was created.
                importance_score = float(
                    item.get("metadata", {}).get(
                        "importance",
                        5
                    )
                )

                # Convert importance from 1-10 to 0-1.
                importance_score = importance_score / 10

                # Calculate how recent the memory is.
                created_at = item.get("created_at")

                recency_score = 0.5

                if created_at:
                    try:
                        from datetime import datetime, timezone

                        created_time = datetime.fromisoformat(
                            created_at.replace(
                                "Z",
                                "+00:00"
                            )
                        )

                        now = datetime.now(timezone.utc)

                        age_days = (
                            now - created_time
                        ).total_seconds() / 86400

                        # Memory loses half its recency score
                        # every 30 days.
                        recency_score = 2 ** (
                            -age_days / 30
                        )

                    except Exception:
                        recency_score = 0.5

                # Final memory ranking.
                item["final_score"] = (
                    relevance_score * 0.60
                    + importance_score * 0.25
                    + recency_score * 0.15
                )

            # Keep only the top 5 memories by final score.
            relevant_memories = sorted(
                relevant_memories,
                key=lambda item: item.get(
                    "final_score",
                    0
                ),
                reverse=True
            )[:5]

            memory_context = "\n".join(
                item["memory"]
                for item in relevant_memories
            )

        # Add user's message to short-term memory.
        self.conversation_history.append({
            "role": "user",
            "text": message
        })

        # Keep only recent conversation.
        recent_history = self.conversation_history[
            -self.max_history:
        ]

        conversation_context = "\n".join(
            f"{item['role']}: {item['text']}"
            for item in recent_history
        )

        # Give Gemini both types of context.
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

        # Add AI response to short-term memory.
        self.conversation_history.append({
            "role": "assistant",
            "text": answer
        })

        # Handle possible memory updates and conflicts.
        if self.check_for_update(message):

            self.update_memory(message)

            # Store the new information if it is worth remembering.
            if self.decide_memory(message):
                self.remember(message, answer)

        else:

            # Store normal long-term information.
            if self.decide_memory(message):
                self.remember(message, answer)

        return answer