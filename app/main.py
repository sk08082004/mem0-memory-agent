from app.agent import Agent
from app.config import USER_ID


def main():
    print("*" * 50)
    print("MEM0 LONG-TERM MEMORY AGENT")
    print("=" * 50)
    print("Type 'exit' to quit.")
    print("Type '/help' to see your commands.\n")

    agent = Agent()

    while True:
        user_input = input("You: ").strip()

        # Exit command
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Show memories command
        if user_input.lower() == "/memories":
            try:
                memories = agent.memory.get_all(USER_ID)

                print("\nStored memories:")

                if memories["count"] == 0:
                    print("No memories stored.")

                else:
                    for i, memory in enumerate(
                        memories["results"],
                        start=1
                    ):
                        print(f"{i}. {memory['memory']}")

                print()

            except Exception as e:
                print(f"\n[ERROR] Could not retrieve memories: {e}\n")

            continue
        # forget memory command
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
                memories = agent.memory.get_all(USER_ID)

                if memory_number < 1 or memory_number > memories["count"]:
                    print("\nInvalid memory number.\n")
                    continue

                memory = memories["results"][memory_number - 1]

                memory_id = memory["id"]

                print(f"\nDeleting: {memory['memory']}")

                agent.memory.delete(memory_id)

                print("Memory deleted.\n")

            except Exception as e:
                print(f"\n[ERROR] Could not delete memory: {e}\n")

            continue

            memories = agent.memory.get_all(USER_ID)

            if memory_number < 1 or memory_number > memories["count"]:
                print("\nInvalid memory number.\n")
                continue

            memory = memories["results"][memory_number -1]

            memory_id = memory["id"]

            print(f"\nDeleting: {memory['memory']}")

            agent.memory.delete(memory_id)

            print("Memory deleted.\n")

            continue

        #Clear all the memoires command
        if user_input.lower() == "/clear":
            try:
                memories = agent.memory.get_all(USER_ID)

                if memories["count"] == 0:
                    print("\nNo memories to clear.\n")
                    continue

                confirm = input(
                    f"\nThis wil delete {memories['count']} memories."
                    "Are you sure? (yes/no): "
                ).strip().lower()

                if confirm == "yes":
                    try:
                        agent.memory.clear(USER_ID)
                        print("\nAll memories deleted.\n")

                    except Exception as e:
                        print(
                            f"\n[ERROR] Could not clear memories: {e}\n"
                        )

                else:
                    print("\nMemory clearing cancelled.")

            except Exception as e:
                print(
                    f"\n[ERROR] Could not retrieve memories: {e}\n"
                )

            continue

        #Help command 
        if user_input.lower() == "/help":
                print("""
            Available commands:

            /memories        View all stored memories. 
            /forget <num>    Delete a specific memory.
            /clear           Delete all memories.
            /new             Start a new conversation.
            /help            Show the help message.
            exit             Exit the agent  
                """)
                continue


        #New conversation command
        if user_input.lower() == "/new":
            agent.conversation_history.clear
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