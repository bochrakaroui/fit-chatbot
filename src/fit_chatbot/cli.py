"""Command-line chat for local smoke testing and offline use."""

from __future__ import annotations

from .service import ChatService


def main() -> None:
    service = ChatService()
    print("Form Fitness Coach. Type 'quit' to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.casefold() in {"quit", "exit"}:
            break
        if message:
            print(f"Form: {service.chat(message).answer}\n")


if __name__ == "__main__":
    main()
