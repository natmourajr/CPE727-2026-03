# Seminário: Processamento de Linguagem Natural

CPE727 - Aprendizado Profundo, PEE/COPPE/UFRJ
Pedro Ivo

Baseado no livro *Dive into Deep Learning* (D2L), capítulos 15 e 16.

```
processamento-linguagem-natural/
├── README.md
├── slides.pdf               slides do seminário
├── main.tex                 fonte dos slides (Beamer, tema DEXPI)
├── secoes/                  um arquivo .tex por tópico dos slides
├── beamer*DEXPI.sty         arquivos do tema
├── reference.bib            referências usadas nos slides
├── img/                     logo e figuras dos slides
└── exemplos/
    ├── textcnn_sentimento.ipynb   exemplo em PyTorch (TextCNN no IMDb)
    └── Dockerfile                 ambiente para rodar o notebook
```

## Conteúdo dos slides

1. Embeddings de palavras: *skip-gram* e CBOW (word2vec)
2. Treinamento aproximado: *negative sampling* e *hierarchical softmax*
3. GloVe
4. *Subword embedding* e analogias
5. BERT: MLM, NSP, função de perda e ajuste fino
6. Análise de sentimento com RNN (BiRNN)
7. Análise de sentimento com CNN (TextCNN)
8. Inferência de linguagem natural com atenção (*decomposable attention*)

## Slides

Compilam com pdfLaTeX. No Overleaf, o compilador padrão funciona.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Exemplo: TextCNN para análise de sentimento

`exemplos/textcnn_sentimento.ipynb` implementa a TextCNN de Kim (2014) em PyTorch
para classificar críticas de filmes em inglês como positivas ou negativas:
tokenização, vocabulário, *embedding*, convoluções 1D com filtros de 3, 4 e 5
palavras, *max-pooling* no tempo e camada de saída. O dataset é o IMDb
(`stanfordnlp/imdb`), baixado pela biblioteca `datasets` (sem cadastro nem chave),
com um subconjunto balanceado de 4.000 críticas de treino e 1.000 de teste.
Em CPU, o treino de 5 épocas leva poucos minutos e chega a cerca de 77% de
acurácia no teste.

### Como executar com Docker

```bash
cd exemplos
docker build -t textcnn-sentimento .
docker run --rm -p 8888:8888 textcnn-sentimento
```

Abra no navegador o link `http://127.0.0.1:8888/?token=...` que aparece no
terminal e execute o notebook. O container precisa de internet na primeira
execução, para baixar o IMDb. Para guardar o cache entre execuções:

```bash
docker run --rm -p 8888:8888 -v hf_cache:/root/.cache/huggingface textcnn-sentimento
```

Para rodar o notebook inteiro sem abrir o Jupyter (útil para testar):

```bash
docker run --rm textcnn-sentimento \
    jupyter nbconvert --to notebook --execute textcnn_sentimento.ipynb --output saida.ipynb
```

### Sem Docker

```bash
pip install torch numpy matplotlib datasets ipywidgets jupyter
jupyter notebook exemplos/textcnn_sentimento.ipynb
```

## Referências

- Zhang, A.; Lipton, Z. C.; Li, M.; Smola, A. J. *Dive into Deep Learning*,
  capítulos 15 e 16. https://d2l.ai
- Mikolov, T. et al. *Efficient Estimation of Word Representations in Vector
  Space*, 2013.
- Pennington, J.; Socher, R.; Manning, C. *GloVe: Global Vectors for Word
  Representation*, EMNLP, 2014.
- Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding*, NAACL, 2019.
- Kim, Y. *Convolutional Neural Networks for Sentence Classification*, EMNLP, 2014.
- Parikh, A. P. et al. *A Decomposable Attention Model for Natural Language
  Inference*, EMNLP, 2016.

As demais referências dos slides estão em `reference.bib`.
