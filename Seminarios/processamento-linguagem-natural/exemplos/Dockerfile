# Ambiente para rodar o textcnn_sentimento.ipynb (PyTorch em CPU + Jupyter)
#
#   docker build -t textcnn-sentimento .
#   docker run --rm -p 8888:8888 textcnn-sentimento
#
# e abrir o link http://127.0.0.1:8888/... que aparece no terminal.
# O notebook baixa o IMDb da Hugging Face na primeira execução, então o
# container precisa de internet. Para guardar o cache entre execuções:
#
#   docker run --rm -p 8888:8888 -v hf_cache:/root/.cache/huggingface textcnn-sentimento

FROM python:3.12-slim

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy matplotlib datasets ipywidgets jupyter

WORKDIR /exemplos
COPY textcnn_sentimento.ipynb ./

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
