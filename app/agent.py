import os
import logging
import json

logging.getLogger("google").setLevel(logging.ERROR)

from google import genai
from google.genai import types

from app.memory import MemoryManager


class Agent:
    """
    AI agent with long-term memory.
    """

    def __init__(self, user_id):
        self.user_id = user_id
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

For every memory, also provide a short reason explaining why the information is worth remembering. 

The reason must be based only on the user's message. 

Return ONLY valid JSON in this exact format:

{{
    "memories": [
        {{
            "text": "standalone factual memory",
            "importance": 8,
            "reason": "why this information is worth remembering"
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
                reason = memory.get("reason", "not specified")

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
                    self.user_id,
                    metadata={
                        "importance": importance, 
                        "reason": reason,
                        "source": "conversation"
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
                self.user_id
            )

        except Exception as e:
            print(
                f"\n[ERROR] Memory search failed: {e}\n"
            )
            return {"results": []}

    def update_memory(self, message, candidate_memories):
        """
        Use Gemini to determine whether the new message changes any
        relevant existing memories and, if so, create the updated versions.

        This runs on every user message. It does not depend on keywords.
        """

        if not candidate_memories:
            return False

        try:
            memory_list = "\n".join(
                f"ID: {memory['id']} | Memory: {memory['memory']}"
                for memory in candidate_memories
            )

            prompt = f"""
You are the memory evolution system for an AI assistant.

Your job is to determine whether the user's NEW message changes,
updates, replaces, or corrects any of the EXISTING memories below.

Do this using semantic understanding, not keywords.
A user may express a change indirectly, briefly, or using a reference
such as "that meeting", "make it 12", or "I prefer the other one".
Use the existing memories and the new message together to understand
what the user means.

Existing relevant memories:

{memory_list}

New user message:

"{message}"

Rules:
1. Return an update only when the new message clearly changes,
   corrects, replaces, or modifies an existing memory.
2. Do not treat a merely related message as an update.
3. Resolve references such as "that", "it", "the meeting", or "the old one"
   using the existing memories when the meaning is clear.
4. When a memory is updated, rewrite it as a complete standalone factual
   statement containing the newest information.
5. Preserve important information from the old memory that is still true.
6. Do not invent facts or details that are not supported by the old memory
   and the new message.
7. If multiple memories are changed, return all of them.
8. If nothing is changed, return an empty updates list.
9. For every updated memory, provide an importance score from 1 to 10
   and a short reason based on the user's new message and the updated fact.

Return ONLY valid JSON in this exact format:

{{
    "updates": [
        {{
            "memory_id": "existing-memory-id",
            "text": "complete updated standalone memory",
            "importance": 8,
            "reason": "why the updated information is worth remembering"
        }}
    ]
}}

If there are no updates, return:

{{
    "updates": []
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

            response_text = response.text.strip()

            if not response_text:
                print("[MEMORY] Gemini returned an empty update response.")
                return False

            if response_text.startswith("```"):
                response_text = response_text.replace("```json", "")
                response_text = response_text.replace("```", "")
                response_text = response_text.strip()

            result = json.loads(response_text)
            updates = result.get("updates", [])

            if not updates:
                return False

            valid_ids = {memory["id"] for memory in candidate_memories}
            updated_any = False

            for update in updates:
                memory_id = update.get("memory_id")
                memory_text = update.get("text", "").strip()
                importance = update.get("importance", 5)
                reason = update.get("reason", "Updated based on the user's latest message.")

                # Never allow Gemini to modify a memory that was not
                # provided as a candidate by our application.
                if memory_id not in valid_ids:
                    continue

                if not memory_text:
                    continue

                try:
                    print(f"[MEMORY] Updating memory: {memory_id}")
                    self.memory.delete(memory_id)

                    messages = [
                        {
                            "role": "user",
                            "content": memory_text
                        }
                    ]

                    self.memory.add(
                        messages,
                        self.user_id,
                        metadata={
                            "importance": importance,
                            "reason": reason,
                            "source": "conversation"
                        }
                    )

                    updated_any = True

                except Exception as e:
                    print(
                        f"\n[ERROR] Could not update memory {memory_id}: {e}\n"
                    )

            return updated_any

        except Exception as e:
            print(
                f"\n[ERROR] Memory update analysis failed: {e}\n"
            )

            if "response" in locals():
                print("[DEBUG] Gemini memory update response:")
                print(response.text)

            return False

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

        # Check every message for possible memory updates.
        # Mem0 finds likely related memories first, then Gemini decides
        # whether the new message actually changes any of them.
        candidate_memories = []

        if memories and "results" in memories:
            candidate_memories = [
                item
                for item in memories["results"]
                if item.get("score", 0) >= 0.15
            ]

            candidate_memories = sorted(
                candidate_memories,
                key=lambda item: item.get("score", 0),
                reverse=True
            )[:10]

        was_updated = self.update_memory(message, candidate_memories)

        # If a memory changed, retrieve again so the response uses the
        # newly stored version instead of the outdated one.
        if was_updated:
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

        # Store normal long-term information only when the message did not
        # update an existing memory. Updated memories have already been
        # replaced by update_memory().
        if not was_updated and self.decide_memory(message):
            self.remember(message, answer)

        return answer