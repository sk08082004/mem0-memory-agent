from app.agent import Agent


def main():
    print("*" * 50)
    print("MEM0 LONG-TERM MEMORY AGENT")
    print("=" * 50)
    print("type 'exit' to quit.\n")

    agent = Agent()

    while True:
        user_input = input("You:").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = agent.respond(user_input)

        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
 

