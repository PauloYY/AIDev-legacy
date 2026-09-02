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

# C, C++, Java, Go, Rust, Ruby e PHP — direto dos repositórios do Debian.
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

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Maven, para builds/testes de projetos Java baseados em pom.xml.
RUN apt-get update \
    && apt-get install -y --no-install-recommends maven \
    && rm -rf /var/lib/apt/lists/*

ENV GRADLE_VERSION=8.10.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && curl -fsSL "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -o /tmp/gradle.zip \
    && unzip -q /tmp/gradle.zip -d /opt \
    && ln -s "/opt/gradle-${GRADLE_VERSION}/bin/gradle" /usr/local/bin/gradle \
    && rm /tmp/gradle.zip \
    && apt-get purge -y --auto-remove curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace