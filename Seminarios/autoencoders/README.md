# Seminário: Autoencoders

CPE727 - Aprendizado Profundo, PEE/COPPE/UFRJ
Alice Cipriano e Leandro Fernandes, 10/11/2026

```
autoencoders/
├── README.md
├── slides.pdf               slides do seminário
├── main.tex                 fonte dos slides (Beamer, tema DEXPI)
├── beamer\*DEXPI.sty         arquivos do tema
├── reference.bib            referências usadas nos slides
├── img/                     imagens do tema
├── figures/
│   ├── Alice/               figuras da primeira parte dos slides
│   ├── Fernandes/           figuras da segunda parte dos slides
│   └── exemplos/            figuras geradas pelo notebook
└── exemplos/
    ├── codigos.ipynb        exemplos em PyTorch
    ├── libs.py              funções do notebook (dados, modelo, treino, figuras)
    └── DockerFile           ambiente para rodar o notebook
```

## Slides

Compilam com pdfLaTeX. No Overleaf, o compilador padrão funciona.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Exemplos

O notebook já está com as saídas, então dá para ler direto no GitHub. Ele usa o conjunto *digits* do scikit-learn: 1.797 imagens 8×8 de dígitos manuscritos, que não precisam ser baixadas. Os resultados que comparam métodos são a média ± desvio padrão de 5 treinamentos com sementes diferentes.

1. Autoencoder $64 \\to 32 \\to 8 \\to 32 \\to 64$ reconstruindo os dígitos.
2. Relação com PCA:

   * o autoencoder linear recupera o subespaço do PCA;
   * o autoencoder não linear segue um arco 2D;
   * erro em função do tamanho do código.
3. Capacidade demais e esparsidade: autoencoder overcomplete com e sem penalidade L1.
4. Profundidade: raso, 2 camadas e 3 camadas com o mesmo código de 8 dimensões.
5. Denoising autoencoder: remoção de ruído e o campo vetorial $g(f(x)) - x$.
6. Contractive autoencoder: valores singulares do jacobiano do encoder.

As figuras são salvas em `figures/exemplos/`. O notebook roda em CPU em cerca de 10 minutos.

Com Docker:

```bash
cd exemplos
docker build -f DockerFile -t autoencoders-exemplos .
docker run --rm -p 8888:8888 autoencoders-exemplos
```

Depois é só abrir o link `http://127.0.0.1:8888/...` que aparece no terminal.

Sem Docker (recomendo usar venv):

```bash
cd exemplos
pip install torch numpy scipy scikit-learn matplotlib jupyter
jupyter notebook codigos.ipynb
```

## Referências

* Goodfellow, Bengio e Courville. *Deep Learning*, cap. 14. MIT Press, 2016. https://www.deeplearningbook.org/contents/autoencoders.html
* Alpaydin, E. e Kaynak, C. *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository, 1998. https://doi.org/10.24432/C50P49
* As demais referências estão em `reference.bib`.

O tema Beamer DEXPI está sob licença CC BY-SA 4.0 (veja o cabeçalho do `main.tex`).

