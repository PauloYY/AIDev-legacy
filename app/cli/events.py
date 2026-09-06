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
        if event.data.get("success", True):
            print("   ✓ concluído")
        else:
            print("   ✗ concluído, mas retornou erro (veja o resultado)")

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

        usage_breakdown = event.data.get("usage_breakdown")

        if usage_breakdown:
            print(f"\n{usage_breakdown}")

    elif event.type == "final_verification_start":
        print("→ Verificação final do projeto...")

    elif event.type == "final_verification_end":
        status = event.data.get("status", "unknown")
        if status == "ok":
            print("   ✓ verificação final: ok")
        elif status == "problems_found":
            print("   ⚠ verificação final: problemas encontrados")
        else:
            print("   ⚠ verificação final: indisponível")

    elif event.type == "final_verification_error":
        print(
            f"   ⚠ erro na verificação final: "
            f"{event.data.get('error', 'erro desconhecido')}"
        )

    elif event.type == "final_verification_warning":
        print(
            f"   ⚠ aviso verificação final: "
            f"{event.data.get('message', '')}"
        )