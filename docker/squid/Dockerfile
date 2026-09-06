# Imagem do proxy de rede restrita do sandbox do AIDev.
#
# Construída localmente (igual à imagem do sandbox em ../Dockerfile),
# em vez de puxar uma imagem de terceiro não verificada — mesma lógica
# de confiança que vocês já aplicam pro resto do projeto.
#
#   docker build -t aidev-sandbox-proxy:latest docker/squid

FROM debian:stable-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends squid \
    && rm -rf /var/lib/apt/lists/*

COPY squid.conf /etc/squid/squid.conf

EXPOSE 3128

CMD ["squid", "-N", "-d", "1"]