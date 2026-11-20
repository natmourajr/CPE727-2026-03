# Seminário: Redes Adversárias Generativas (GANs)

CPE727 - Aprendizado Profundo, PEE/COPPE/UFRJ
Pedro Henrique Batista Campos e Daniel Cordeiro Marques, 12/11/2026

Seminário sobre o capítulo 20 de *Deep Learning* (Goodfellow, Bengio e Courville), com a didática do curso de
GANs da DeepLearning.AI: como uma GAN funciona (o jogo entre gerador e discriminador e o ótimo teórico), a DCGAN
e os artefatos de tabuleiro, os problemas da dinâmica adversarial (colapso de modos, vanishing gradient,
Wasserstein e WGAN-GP), como controlar e avaliar o que ela gera (GAN condicional, FID, precisão e cobertura),
StyleGAN, Pix2Pix e CycleGAN, e as aplicações. Fecha com um exemplo nosso, treinado do zero nos sprites de
Pokémon do *Dive into Deep Learning*: as mesmas perdas lado a lado e uma StyleGAN que mistura espécies.

```
gan/
├── README.md
├── slides.pdf          slides do seminário
├── main.tex            fonte dos slides (Beamer)
├── coppecolor.sty      paleta de cores da COPPE usada no main.tex
├── referencias.bib     referências dos slides e dos exemplos
├── figs/               figuras usadas no main.tex
├── logos/              logos da COPPE e do PEB
└── exemplos/
    ├── critico_1d.py   D (BCE) contra crítico (WGAN-GP) em 1D: o sinal que chega ao gerador
    └── pokemon/        DCGAN em seis perdas e StyleGAN no Pokémon do D2L, com Dockerfile
```

## Slides

O PDF é a versão apresentada. O `main.tex` compila com pdfLaTeX, a partir desta pasta (as figuras são achadas
por `\graphicspath`, e o `coppecolor.sty` fica ao lado). Usa a fonte Carlito (pacote `carlito`, já incluído no
TeX Live e no Overleaf).

```bash
pdflatex main.tex
pdflatex main.tex
```

A segunda passada acerta o total de páginas no rodapé. No Overleaf, basta enviar esta pasta com o compilador padrão.

As referências estão no rodapé de cada slide, escritas direto no `main.tex`; o `referencias.bib` reúne as
mesmas obras em BibTeX. Os diagramas e gráficos marcados como "nosso" foram feitos por nós; as demais figuras
vêm do curso da DeepLearning.AI e dos artigos citados em cada slide.

## Exemplos

### Crítico em 1D (`exemplos/critico_1d.py`)

Reproduz a ideia da figura 2 de Arjovsky et al. (2017): com a distribuição falsa parada longe da real, o
discriminador treinado até o fim satura e o sinal da perda minimax some onde estão as amostras falsas, enquanto o
crítico da WGAN-GP continua inclinado. Roda em CPU em menos de um minuto e grava `resultados/critico_1d.json`.

```bash
cd exemplos
pip install torch numpy
python critico_1d.py
```

### GANs no Pokémon (`exemplos/pokemon/`)

Código do exemplo dos últimos slides, também publicado em
[github.com/ph-b-campos/poke_gan](https://github.com/ph-b-campos/poke_gan):

- **DCGAN** com a mesma arquitetura em vários modos de treino: minimax (vanishing gradient), não saturante,
  D pequeno (colapso de modos), WGAN com clipping, WGAN-GP e WGAN-GP condicional na espécie (com e sem
  mistura de estilos).
- **StyleGAN-like** (`models/stylegan`) para o mix de domínio entre espécies: a forma de uma, as cores de outra.
  Nos slides ele aparece como `exemplos/pokemon/estilo_gan.py`, o nome do script antes da organização em pastas;
  hoje o modelo está em `models/stylegan/` e o treino em `training/stylegan.py`.
- **Juiz**: um classificador de espécies treinado nos sprites reais, usado para o FID e para contar espécies.

O código está separado em `models/` (cada modelo com sua pasta `config/` de hiperparâmetros), `training/` e
`analise/` (métricas e figuras). Todos os comandos rodam a partir de `exemplos/pokemon`:

```bash
cd exemplos/pokemon
pip install -r requirements.txt
python -m data_handler --baixar
python -m training.juiz
python -m training.dcgan --modo wgp
python -m training.stylegan --modo cor --horas 8
python -m analise.figuras_pokemon curvas
python -m analise.figuras_estilo --modo cor --pares 25:249 --so-misturas
```

Nessa ordem: baixa os sprites do D2L e monta o cache 64x64 em `dados/`, treina o juiz, treina as GANs e gera as
figuras em `resultados/figs/`. O treino é pensado para GPU (a StyleGAN dos slides levou 5,6 h numa RTX 2060);
em CPU serve para gerar figuras de um modelo já treinado.

Com Docker, a partir de `exemplos/pokemon` (no Linux com GPU NVIDIA, acrescente `--gpus all` ao `docker run`):

```bash
docker build -t poke-gan .
docker run --rm -v "$PWD/dados:/app/dados" -v "$PWD/resultados:/app/resultados" poke-gan
docker run --rm -v "$PWD/dados:/app/dados" -v "$PWD/resultados:/app/resultados" poke-gan python -m training.juiz
```

O primeiro `docker run` baixa os dados; os demais passos seguem a mesma forma, trocando o comando final.
O README de [poke_gan](https://github.com/ph-b-campos/poke_gan) descreve todas as opções.

## Referência principal

Goodfellow, Bengio e Courville. Deep Learning, cap. 20 (Deep Generative Models). MIT Press, 2016.
https://www.deeplearningbook.org/contents/generative_models.html
