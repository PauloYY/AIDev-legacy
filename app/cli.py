from app.agent.events import AgentEvent


def handle_agent_event(event: AgentEvent):
    if event.type == "agent_start":
        print("* AIDev está trabalhando...\n")

    elif event.type == "tool_start":
        name = event.data["name"]
        arguments = event.data["arguments"]

        if name == "list_files":
            project = arguments.get("project_name")
            print(f"list_files → {project}")

        elif name == "read_file":
            file_path = arguments.get("file_path")
            print(f"read_file → {file_path}")

        elif name == "write_file":
            file_path = arguments.get("file_path")
            print(f"write_file → {file_path}")

        else:
            print(f"{name}")

    elif event.type == "tool_end":
        print("   ✓ concluído")

    elif event.type == "agent_done":
        print("\nAIDev finalizou.")