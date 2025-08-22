# -----------------------------
# Dockerfile para classificação de imagens com ResNet50
# -----------------------------

# Imagem base
FROM python:3.11-slim

# Evitar buffers e reduzir logs do TensorFlow
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar arquivos do projeto para o container
COPY model.py requirements.txt ./

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip e instalar dependências do Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Criar pastas para dataset e resultados
RUN mkdir -p /app/dataset
RUN mkdir -p /app/results

# Comando padrão para rodar o script
CMD ["python", "model.py"]
