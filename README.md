# Mem0 Long-Term Memory Agent

A local long-term memory AI assistant built with Python, Google Gemini, Mem0, a Streamlit dashboard, and a CLI for diagnostics and administration.

## Overview

This project extends Gemini with a long-term memory layer so the agent can remember important facts, update them over time, and retrieve relevant context in future conversations. It is designed for interactive use from the terminal and stores persistent memory in Mem0 while keeping a short local conversation window for immediate context.

## Tech Stack

- Python
- Google Gemini (`gemini-3.5-flash-lite`)
- Mem0 Cloud / Mem0 MemoryClient
- `google-genai`
- `python-dotenv`
- CLI-based workflow
- Streamlit dashboard for conversation and read-only monitoring

## Dashboard

Install the dependencies from `requirements.txt`, configure `GEMINI_API_KEY`
and `MEM0_API_KEY`, then start the local dashboard from the repository root:

```powershell
streamlit run app/dashboard.py
```

Conversation is the dashboard's default page. It calls the existing
`Agent.respond()` pipeline, so memory retrieval, updates, and relationship
safeguards remain agent-managed. Conversation transcripts are stored in the
git-ignored local file `data/conversations.json`, scoped to the selected local
user, and written atomically. This is local persistence, not cloud storage or
authentication. The transcript is separate from long-term Mem0 facts;
starting or deleting a conversation does not clear or mutate memories. The
sidebar searches conversation titles and messages and supports switching,
renaming, and deleting transcripts. If a local write fails, the current
browser session keeps the unsaved conversation visible and retries the save
when storage becomes available; keep the page open until the warning clears.
The existing CLI remains available for diagnostics and administrative
operations.

Technical commands remain available directly in the conversation composer.
`/help` uses the same command list in the dashboard and CLI. Commands run
locally rather than being sent to Gemini, and command output is marked
separately from normal assistant responses. Destructive `/clear` asks for
confirmation in the dashboard; `exit` keeps its CLI meaning and does not stop
the Streamlit app. Natural-language messages continue through the regular
agent response pipeline.

## Current architecture

The current agent uses:

- Gemini for response generation
- Gemini for deciding whether information should become long-term memory
- Gemini for extracting memories
- Gemini for assigning importance and confidence
- Gemini for semantic deduplication and memory evolution decisions
- Mem0 for persistent long-term semantic memory storage and retrieval
- a local rolling conversation history for recent context

The system stores short-term context in-memory within the running agent session and keeps the latest 10 conversation items in the active window.

## How it works

1. The user sends a message.
2. Relevant long-term memories are retrieved semantically from Mem0.
3. The agent checks whether the message updates an existing memory using semantic understanding.
4. Gemini generates a response using both long-term memory and recent conversation context.
5. Gemini decides whether the message contains something worth remembering.
6. If appropriate, the agent extracts standalone memories, assigns importance and confidence, and stores them in Mem0.
7. The system deduplicates similar memories and preserves historical versions in the memory timeline when a fact changes.

## Features

### Short-term conversation memory

The agent keeps a rolling conversation history using a limited recent-message window.

- Current implementation keeps the latest 10 conversation items.
- Recent conversation is always included as part of the model context.

### Persistent long-term semantic memory

Important information is stored in Mem0 and persists across application restarts.

- Long-term memories are retrieved semantically, not by simple keyword matching.
- Stored memories remain available for future sessions and user interactions.

### Mem0 event and graph synchronization reliability

Each extracted fact remains an independent Mem0 memory. Mem0 writes are
followed by bounded event polling; the defaults are an 8-second polling
deadline, a 250 ms initial interval, and a 1-second maximum backoff interval.
Set `MEM0_EVENT_TIMEOUT_SECONDS`,
`MEM0_EVENT_POLL_INTERVAL_SECONDS`, or
`MEM0_EVENT_POLL_MAX_INTERVAL_SECONDS` to tune those values. Event HTTP
requests use the remaining polling time as their request timeout.

When Mem0 accepts a write but its result cannot yet be verified, the agent
preserves the original event ID and memory text in the per-user graph operation
ledger. It does not blindly repeat that write or create a graph node under an
unverified ID. Use `/reconcile-memory-events [limit]` in the CLI or dashboard
chat to recheck pending events and synchronize confirmed memories. A confirmed
Mem0 memory whose graph node or relationships are still pending is reported as
partial success rather than full synchronization. Polling attempts, sleep
duration, Mem0 writes, graph persistence, and relationship generation are
included in the response timing metadata.

Graph relationships use stable Mem0 IDs, an allowlist of directed labels, and
exact text evidence from both endpoint memories. Generic `related_to` edges,
missing endpoints, and duplicate directed edges are rejected or omitted.
Relationship repairs are written as atomic edge deltas with a per-user revision
journal; `KnowledgeGraph.rollback_relationship_revision(user_id, revision_id)`
can restore the prior edges without changing memory nodes.

### Intelligent memory decisions

The agent does not rely on a hardcoded list of keywords to decide what should be remembered.

Gemini evaluates whether the information is likely to remain useful in future conversations, including:

- temporary observations
- meaningful context
- future plans
- future events
- deadlines and appointments
- ongoing projects
- preferences
- important context about people, organizations, or topics

This includes time-sensitive information such as future events and plans when they may matter later.

### Memory extraction

Gemini extracts useful standalone memories from user messages.

Each extracted memory can include:

- `text`
- `importance`
- `confidence`
- `reason`

The system can extract important facts from larger or more complex messages instead of relying on a single sentence only.

### Importance scoring

Each memory receives an importance score from 1 to 10.

- 1-3: minor information
- 4-6: moderately useful context
- 7-8: important long-term information
- 9-10: highly valuable future context

Importance reflects how useful the information is for future conversations, separate from confidence.

### Memory confidence

Each memory receives a confidence score from 0 to 1.

- higher confidence: definite or strongly supported statements
- medium confidence: tentative or qualified statements
- lower confidence: speculative or uncertain statements

Confidence is stored in Mem0 metadata and is accessible through `/memory-info`.

### Memory provenance

Each memory stores provenance metadata, including:

- source
- reason
- creation timestamp
- update timestamp

The current implementation records the source as:

- `conversation`

The reason is generated by Gemini based on why the information is worth remembering.

### Semantic memory deduplication

The project performs Gemini-based semantic deduplication before storing a new memory.

The flow is:

1. Search for potentially similar memories.
2. Filter candidates by semantic relevance.
3. Ask Gemini whether the new memory is a true duplicate.
4. Merge true duplicates.
5. Delete the duplicated memory.
6. Keep distinct facts separate when they are related but not the same fact.

The project-level deduplication threshold is `0.35` for candidate selection during deduplication analysis.

This threshold is application-specific to this project and is not described as a universal Mem0 requirement.

### Memory evolution and timeline tracking

The system supports evolving memories.

When a fact changes, the agent does not show every old version as a separate active user-facing memory. Instead:

- the latest version becomes the active memory
- prior versions remain in the memory's evolution history
- the timeline preserves the change chain

Example timeline:

- `3 PM -> 12 PM -> 10 AM`

The user-facing `/memories` list shows the current version, not each historical state. The full timeline is available through `/memory-info`.

This update system supports repeated changes and keeps the full chain of previous states.

The project-level evolution candidate threshold is `0.25` when analyzing related memories for updates.

### Contextual memory updates

The memory evolution pipeline uses semantic understanding instead of keyword-only comparison.

It can interpret contextual updates such as:

- "Move my meeting to 12 PM."
- "Move it to 3 PM."
- "Change that to tomorrow."
- "Update the old one."

The agent uses existing memory context to resolve references and decide whether a memory should be updated.

### Historical memory protection

Once an evolving memory has older versions, those historical states are filtered out so they are not treated as separate active memories.

This protection is applied during:

- update candidate selection
- normal long-term memory retrieval context

This prevents an older timeline item from being updated again by accident.

### Memory retrieval and ranking

The system retrieves memories and ranks them using a combined score based on:

- semantic relevance
- importance
- recency

The current ranking logic combines these signals in the following way:

- semantic relevance contributes 60%
- importance contributes 25%
- recency contributes 15%

Relevant memories are filtered and then sorted before being included in the response context.

## CLI commands

The project is a terminal-first agent. The command set in the current implementation is:

| Command | Description |
|---|---|
| `/memories` | Show the current stored memories |
| `/memory-info <num>` | Show detailed memory metadata, provenance, and timeline |
| `/forget <num>` | Delete a specific memory |
| `/clear` | Delete all memories for the current user |
| `/new` | Start a new conversation while keeping long-term memory |
| `/user` | Show the currently active user ID |
| `/login <user_id>` | Switch to another user |
| `/logout` | Log out and choose a new user |
| `/help` | Display the command list |
| `exit` | Quit the agent |

## Project structure

```text
mem0-memory-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── main.py
│   └── memory.py
├── data/
├── .env
├── README.md
├── requirements.txt
├── session.json
└── future work.txt
```

## Setup

```bash
git clone <YOUR_REPOSITORY_URL>
cd mem0-memory-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file with the required API keys:

```env
GEMINI_API_KEY=your_google_gemini_api_key
MEM0_API_KEY=your_mem0_api_key
```

Then run:

```bash
python -m app.main
```

On first launch, the agent prompts for a user ID and stores it in `session.json` so the same user can be reused on subsequent runs.

## Primary operations dashboard

The read-only dashboard is the primary interface for monitoring and exploring
Mem0 memory and relationship-recovery state. Launch it from the repository
root:

```bash
streamlit run app/dashboard.py
```

Enter the local user ID and graph file path in the sidebar, then load the
snapshot. The dashboard reads only the selected user's graph from the existing
JSON file. It does not call Mem0 or Gemini and does not modify graph, session,
or recovery data. Recovery actions remain available through the CLI.

The user ID field scopes a local file lookup; it is **not authentication**.
Do not expose the dashboard to a public network until sign-in and server-side
authorization are in place.

## Deployment and persistence status

The application is not cloud-ready. The CLI still persists its active user in
`session.json` and graph nodes, relationships, and recovery records in
`graph.json`. The dashboard currently reads the JSON graph through a
provider-style snapshot interface so a PostgreSQL-backed read provider can be
introduced without embedding storage parsing in the UI. This does not migrate
the CLI or establish a cloud source of truth.

The existing Mem0 Cloud and Gemini integrations remain external services;
their API keys must be supplied through environment variables or deployment
secrets and must never be committed.

## Notes

- Use the dashboard as the primary view for memory, graph, recovery, and
  integrity status. Use the command-line agent for memory and recovery
  operations that change data.
- The dashboard is read-only and reports historical performance as unavailable
  because operation timings are not persisted.
