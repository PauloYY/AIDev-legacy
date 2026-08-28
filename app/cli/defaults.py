"""Objetivo de demonstração usado quando nenhum --objective é informado."""

DEFAULT_PROJECT_NAME = "habit_tracker"

DEFAULT_PROMPT = """
Crie um programa python de gerenciamento de hábitos para ser executado no terminal.

O programa deve iniciar com um menu interativo.

O usuário deve poder:

* cadastrar um novo hábito;
* listar todos os hábitos;
* marcar um hábito como concluído no dia atual;
* visualizar o histórico de conclusão de um hábito;
* remover um hábito;
* visualizar a sequência atual de dias consecutivos em que um hábito foi concluído.

Os dados devem ser persistidos em um arquivo local para que não sejam perdidos quando o programa for encerrado.

Regras:

* um hábito não pode ser marcado como concluído duas vezes no mesmo dia;
* cada conclusão deve registrar a data;
* a sequência de dias consecutivos deve ser calculada com base no histórico de conclusões;
* o programa deve tratar entradas inválidas e hábitos inexistentes adequadamente.

Organize o projeto em módulos com responsabilidades bem definidas.

Use a pasta `habit_tracker` para o projeto.

Mantenha o projeto simples e fácil de entender.
"""
