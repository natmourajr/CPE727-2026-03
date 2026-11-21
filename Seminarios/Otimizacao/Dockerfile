FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# PyTorch CPU: roda sem GPU e sem necessidade de Google Colab.
COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0 torchvision==0.25.0 && \
    pip install --no-cache-dir "numpy>=1.26,<3" "matplotlib>=3.8,<4" "pillow>=10,<13"

COPY seminario_8_1.py seminario_completo.py config.json ./
COPY capitulos/ ./capitulos/
RUN mkdir -p /app/graficos /app/dados

VOLUME ["/app/graficos", "/app/dados"]
CMD ["python", "seminario_completo.py", "--chapters", "all", "--config-8-1", "config.json", "--outdir", "/app/graficos"]
