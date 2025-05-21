# V4_API/Dockerfile
# Dockerfile usando Debian Slim como base
FROM python:3.10-slim-bullseye

# Usar um usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Instalar dependências essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Baixar recursos NLTK necessários
RUN python -m nltk.downloader vader_lexicon punkt stopwords wordnet

# Criar diretório para dados NLTK com permissões corretas
RUN mkdir -p /app/nltk_data && \
    cp -r /root/nltk_data/* /app/nltk_data/ && \
    chown -R appuser:appuser /app/nltk_data

# Configurar variável de ambiente para NLTK usar o diretório correto
ENV NLTK_DATA=/app/nltk_data

# Copiar o código do projeto
COPY . .

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dar permissões para o usuário appuser
RUN chown -R appuser:appuser /app

# Trocar para usuário não-root
USER appuser

# Expor a porta que o serviço usará
EXPOSE 8080

# Iniciar o serviço
CMD ["gunicorn", "app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]