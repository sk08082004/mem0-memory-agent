from app.agent import Agent
from app.config import SESSION_FILE
import json, os


def load_user():
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

            return data.get("user_id")

    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_user(user_id):
    with open(SESSION_FILE, "w", encoding="utf-8") as file:
        json.dump({"user_id": user_id}, file)


def remove_user():
    try:
        os.remove(SESSION_FILE)
    except FileNotFoundError:
        pass


def get_active_memories(agent):
    """Return only current memories, hiding historical timeline versions."""
    memories = agent.memory.get_all(agent.user_id)
    results = memories.get("results", [])

    historical_ids = set()

    for memory in results:
        evolution = memory.get("metadata", {}).get("evolution")

        if isinstance(evolution, str):
            try:
                evolution = json.loads(evolution)
            except (json.JSONDecodeError, TypeError):
                evolution = None

        if isinstance(evolution, dict):
            previous_id = evolution.get("previous_memory_id")
            if previous_id:
                historical_ids.add(previous_id)

            history = evolution.get("history", [])
            if isinstance(history, list):
                for change in history:
                    if isinstance(change, dict):
                        memory_id = change.get("memory_id")
                        if memory_id:
                            historical_ids.add(memory_id)

        elif isinstance(evolution, list):
            for item in evolution:
                if isinstance(item, str) and item.startswith("previous_memory_id."):
                    historical_ids.add(item.split(".", 1)[1])

    active_results = [
        memory
        for memory in results
        if memory.get("id") not in historical_ids
    ]

    return active_results


def main():
    print("*" * 50)
    print("MEM0 LONG-TERM MEMORY AGENT")
    print("=" * 50)
    print("Type 'exit' to quit.")
    print("Type '/help' to see your commands.\n")

    user_id = load_user()

    if not user_id:
        user_id = input("Enter your user ID: ").strip()
        save_user(user_id)

    agent = Agent(user_id)

    while True:
        user_input = input("You: ").strip()

        # Exit command
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Show memories command
        if user_input.lower() == "/memories":
            try:
                active_memories = get_active_memories(agent)

                print("\nStored memories:")

                if not active_memories:
                    print("No memories stored.")
                else:
                    for i, memory in enumerate(
                        active_memories,
                        start=1
                    ):
                        print(f"{i}. {memory['memory']}")

                print()

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memories: {e}\n"
                )

            continue

        # Show detailed information about a memory
        if user_input.lower().startswith("/memory-info"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /memory-info <number>\n")
                continue

            try:
                memory_number = int(parts[1])
            except ValueError:
                print("\nPlease enter a valid memory number.\n")
                continue

            try:
                active_memories = get_active_memories(agent)

                if (
                    memory_number < 1
                    or memory_number > len(active_memories)
                ):
                    print("\nInvalid memory number.\n")
                    continue

                memory = active_memories[memory_number - 1]
                metadata = memory.get("metadata", {})

                print("\nMemory details:")
                print(f"Memory: {memory['memory']}")
                print(
                    f"Importance: "
                    f"{metadata.get('importance', 'N/A')}"
                )
                print(
                    f"Reason: "
                    f"{metadata.get('reason', 'N/A')}"
                )
                print(
                    f"Source: "
                    f"{metadata.get('source', 'N/A')}"
                )
                print(
                    f"Created: "
                    f"{memory.get('created_at', 'N/A')}"
                )
                print(
                    f"Updated: "
                    f"{memory.get('updated_at', 'N/A')}"
                )

                evolution = metadata.get("evolution")

                if isinstance(evolution, str):
                    try:
                        evolution = json.loads(evolution)
                    except (json.JSONDecodeError, TypeError):
                        evolution = None

                history = []

                if isinstance(evolution, dict):
                    history = evolution.get("history", [])

                if history:
                    print("\nTimeline:")
                    print(
                        f"  Originally: "
                        f"{history[0].get('from', 'N/A')}"
                    )

                    for change in history:
                        print(
                            f"  Updated to: "
                            f"{change.get('to', 'N/A')}"
                        )

                print()

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memory details: {e}\n"
                )

            continue

        # Show current user
        if user_input.lower() == "/user":
            print(f"\nCurrent user: {agent.user_id}\n")
            continue

        # Login as another user
        if user_input.lower().startswith("/login"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /login <user_id>\n")
                continue

            user_id = parts[1].strip()

            if not user_id:
                print("\nUser ID cannot be empty.\n")
                continue

            save_user(user_id)
            agent = Agent(user_id)

            print(f"\nLogged in as: {agent.user_id}\n")
            continue

        # Logout current user
        if user_input.lower() == "/logout":
            remove_user()

            user_id = input("Enter your user ID: ").strip()

            if not user_id:
                print("\nUser ID cannot be empty.\n")
                continue

            save_user(user_id)
            agent = Agent(user_id)

            print(f"\nLogged in as: {agent.user_id}\n")
            continue

        # Forget memory command
        if user_input.lower().startswith("/forget"):
            parts = user_input.split()

            if len(parts) != 2:
                print("\nUsage: /forget <number>\n")
                continue

            try:
                memory_number = int(parts[1])
            except ValueError:
                print("\nPlease enter a memory number.\n")
                continue

            try:
                active_memories = get_active_memories(agent)

                if (
                    memory_number < 1
                    or memory_number > len(active_memories)
                ):
                    print("\nInvalid memory number.\n")
                    continue

                memory = active_memories[memory_number - 1]
                memory_id = memory["id"]

                print(f"\nDeleting: {memory['memory']}")

                agent.memory.delete(memory_id)

                print("Memory deleted.\n")

            except Exception as e:
                print(
                    f"\n[ERROR] Could not delete memory: {e}\n"
                )

            continue

        # Clear all memories command
        if user_input.lower() == "/clear":
            try:
                memories = agent.memory.get_all(agent.user_id)

                if memories["count"] == 0:
                    print("\nNo memories to clear.\n")
                    continue

                confirm = input(
                    f"\nThis will delete {memories['count']} memories. "
                    "Are you sure? (yes/no): "
                ).strip().lower()

                if confirm == "yes":
                    try:
                        agent.memory.clear(agent.user_id)
                        print("\nAll memories deleted.\n")

                    except Exception as e:
                        print(
                            f"\n[ERROR] Could not clear memories: {e}\n"
                        )

                else:
                    print("\nMemory clearing cancelled.\n")

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memories: {e}\n"
                )

            continue

        # Help command
        if user_input.lower() == "/help":
            print("""
Available commands:

/memories              View all stored memories.
/memory-info <num>     Show detailed memory provenance.
/forget <num>          Delete a specific memory.
/clear                 Delete all memories.
/new                   Start a new conversation.
/user                  Show current user.
/login <user_id>       Switch to another user.
/logout                Logout and choose another user.
/help                  Show the help message.
exit                   Exit the agent
            """)
            continue

        # New conversation command
        if user_input.lower() == "/new":
            agent.conversation_history.clear()
            print("\nNew conversation started.\n")
            continue

        # Ignore empty input
        if not user_input:
            continue

        # Normal conversation
        response = agent.respond(user_input)

        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()