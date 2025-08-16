# Imagem base enxuta com Python 3.11
FROM python:3.11-slim

# Variáveis de ambiente para reduzir logs do TF e matplotlib headless
ENV TF_CPP_MIN_LOG_LEVEL=2  
#0=all, 1=info, 2=warnings, 3=errors
ENV MPLBACKEND=Agg

# Diretório de trabalho
WORKDIR /app

# Criar pasta de logs
RUN mkdir -p /app/logs

# Instalar dependências do sistema (necessárias para algumas libs como matplotlib)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar restante do código
COPY . .

# Comando padrão
CMD ["python", "model.py"]
