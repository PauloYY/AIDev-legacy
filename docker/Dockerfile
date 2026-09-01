FROM python:3.12-slim

# --- Runtimes disponíveis no sandbox ---
# O container roda com --network none (sem acesso à internet), então
# qualquer runtime/dependência que o código gerado precise TEM que estar
# pré-instalado aqui em build time. Adicione mais runtimes conforme os
# projetos que o agente for gerar precisarem.

# Python (já vem na imagem base) + pytest, para validar projetos Python.
RUN pip install --no-cache-dir pytest==9.1.1

# Node.js LTS + npm, para validar projetos JavaScript/TypeScript.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# C, C++, Java, Go, Rust, Ruby e PHP
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        default-jdk-headless \
        golang-go \
        rustc \
        cargo \
        ruby \
        php-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace