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

        prompt = f"""
Extract the important long-term information from the user's message.

The information may be about anything. Do not use a fixed list of
categories.

Keep information that is likely to remain useful in future conversations,
including important context, ongoing projects, ideas, plans, experiences,
relationships, preferences, and substantial explanations.

Remove temporary details that are only relevant to the current moment.

If the user provides a detailed explanation of a project, idea, system,
research, workflow, or other ongoing subject, preserve the important
information needed to understand that subject later.

Do not invent information.
Only extract information explicitly supported by the user's message.

Return a concise list of standalone factual memories.
Return ONLY the memories, one per line.

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

            memory_text = extraction.text.strip()

            if not memory_text:
                return None

            messages = [
                {
                    "role": "user",
                    "content": memory_text
                }
            ]

            return self.memory.add(
                messages,
                USER_ID
            )

        except Exception as e:
            print(f"\n[ERROR] Memory extraction/storage failed: {e}\n")
            return None

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

            return result.get("should_remember", False)

        except Exception as e:
            print(f"\n[ERROR] Gemini memory decision failed: {e}\n")
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