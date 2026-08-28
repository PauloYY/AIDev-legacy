from app.agent.events import AgentEvent


def handle_agent_event(event: AgentEvent):
    """Callback de UI: imprime o progresso do agente no terminal.

    Isto é output voltado ao usuário (progresso da execução), diferente do
    logging interno configurado em app.logging_config — por isso continua
    usando print() em vez de logging.
    """

    if event.type == "agent_start":
        print("* AIDev está trabalhando...\n")

    elif event.type == "planner_start":
        print("→ Planner analisando...")

    elif event.type == "planner_end":
        print(
            f"   ✓ decisão: "
            f"{event.data['action']}"
        )

    elif event.type == "planner_error":
        print(
            f"   ⚠ erro no Planner: "
            f"{event.data['error']}"
        )
        print("   ↻ tentando novamente...")

    elif event.type == "executor_start":
        print(
            f"→ Executor analisando: "
            f"{event.data['tool']}"
        )

    elif event.type == "executor_end":
        print(
            f"   ✓ execução planejada: "
            f"{event.data['tool']}"
        )

    elif event.type == "executor_error":
        print(
            f"   ⚠ erro no Executor: "
            f"{event.data['error']}"
        )
        print("   ↻ tentando novamente...")

    elif event.type == "tool_start":
        name = event.data["name"]
        arguments = event.data["arguments"]

        if name == "list_files":
            project = arguments.get("project_name")
            print(
                f"list_files → {project}"
            )

        elif name == "read_file":
            file_path = arguments.get("file_path")
            print(
                f"read_file → {file_path}"
            )

        elif name == "write_file":
            file_path = arguments.get("file_path")
            print(
                f"write_file → {file_path}"
            )

        elif name == "find_references":
            symbol = arguments.get("symbol")
            print(
                f"find_references → {symbol}"
            )

        else:
            print(
                f"{name} → {arguments}"
            )

    elif event.type == "tool_end":
        print("   ✓ concluído")

    elif event.type == "tool_error":
        print(
            f"   ✗ erro: "
            f"{event.data['error']}"
        )

    elif event.type == "agent_error":
        print(
            f"\n* Erro no agente: "
            f"{event.data['error']}"
        )

    elif event.type == "agent_done":
        print("\nAIDev finalizou.")

        usage = event.data.get("usage")

        if usage:
            print(f"\n{usage}")
