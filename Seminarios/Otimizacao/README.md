# Seminário de otimização — Goodfellow 8.1 e 8.4–8.7

Projeto com experimentos e visualizações em **PyTorch CPU**, sem Google Colab, datasets externos ou GPU. O arquivo original `seminario_8_1.py`, obtido do notebook do usuário, foi **preservado**. Os capítulos novos estão organizados em `capitulos/` e podem ser executados separadamente.

Fonte: https://www.deeplearningbook.org/contents/optimization.html

> **Escopo:** contém **8.1.1–8.1.3** + **8.4–8.7.6**. As seções 8.2 e 8.3 não fazem parte deste pacote. Experimentos didáticos; não constituem benchmarks nem reproduções exatas de figuras do livro.

## Arquivos principais

- `seminario_8_1.py`: conteúdo convertido do Colab; experimentos de 8.1.
- `seminario_completo.py`: executa todos os capítulos, alguns capítulos ou uma subseção 8.7.x.
- `capitulos/cap_8_4.py`: estratégias de inicialização de parâmetros (zeros/normal/Xavier/Kaiming).
- `capitulos/cap_8_5.py`: AdaGrad, RMSProp, Adam; trajetórias, taxas de aprendizagem efetivas e comparação em classificação.
- `capitulos/cap_8_6.py`: Newton (Hessiana via autograd), gradientes conjugados e BFGS clássico; trajetórias e exemplo de ponto de sela.
- `capitulos/cap_8_7.py`: BatchNorm; descida por coordenadas; Polyak averaging; pré-treinamento supervisionado; conexões residuais; curriculum learning.
- `capitulos/comum.py`: geração reprodutível de dados e utilitários de gráficos.
- `tests/`: testes de fumaça das novas seções.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` e `config.json`.

## Docker (Windows / PowerShell)

No terminal da pasta extraída, com Docker Desktop em execução:

```powershell
docker compose up --build
```

Isso executa 8.1, 8.4, 8.5, 8.6 e 8.7. Todas as figuras são gravadas em `./graficos/` no seu computador. A pasta contém também `figuras_todas.zip` e `relatorio_execucao.json`.

Para executar **somente os novos capítulos** sem repetir os experimentos 8.1:

```powershell
docker compose run --rm seminario python seminario_completo.py --chapters 8.4,8.5,8.6,8.7 --outdir /app/graficos
```

Para um teste rápido:

```powershell
docker compose run --rm seminario python seminario_completo.py --chapters 8.4,8.5,8.6,8.7 --quick --outdir /app/graficos
```

Para uma subseção específica:

```powershell
docker compose run --rm seminario python seminario_completo.py --chapters 8.7 --subsection 8.7.3 --outdir /app/graficos
```

> Primeira execução com `docker compose run` também compila a imagem, caso ainda não exista. O `Dockerfile` usa wheels PyTorch CPU e requer acesso à internet apenas para instalar pacotes na etapa de build.

## Python local (sem Docker)

Python 3.11 é recomendado. Exemplo PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe seminario_completo.py --chapters 8.4,8.5,8.6,8.7 --outdir graficos
```

Recomendação CPU e poucos threads (`OMP_NUM_THREADS=2`) em computadores com poucos recursos. Para 8.1 use o `seminario_completo.py` sem `--config-8-1`, caso rode localmente, pois o `config.json` fornecido contém caminhos `/app` do contêiner.

## Índice dos gráficos por seção

| Seção | Experimento | Saídas principais |
|---|---|---|
| 8.4 | inicialização zero, normal, Xavier e Kaiming | `8_4_01`, `8_4_02` |
| 8.5.1–8.5.3 | AdaGrad, RMSProp e Adam | `8_5_01`, `8_5_02`, `8_5_03` |
| 8.5.4 | comparação de otimizadores em classificação | `8_5_04` |
| 8.6.1–8.6.3 | GD, Newton, CG e BFGS | `8_6_01`, `8_6_02`, `8_6_03` |
| 8.7.1 | Batch Normalization | `8_7_1_batch_normalization.png` |
| 8.7.2 | Coordinate Descent | `8_7_2_*.png` |
| 8.7.3 | Polyak Averaging | `8_7_3_*.png` |
| 8.7.4 | Supervised Pretraining | `8_7_4_*.png` |
| 8.7.5 | Designing Models to Aid Optimization | `8_7_5_*.png` |
| 8.7.6 | Curriculum Learning | `8_7_6_*.png` |

## Cuidados na interpretação

- As curvas dependem de hiperparâmetros, inicialização e amostragem; resultados podem mudar com a semente.
- Para estudar convergência, observe tanto o número de **atualizações** quanto o custo de computação; não assuma que um algoritmo é sempre o melhor.
- CG, Newton e BFGS usam **quadráticas pequenas** para permitir visualizar a matemática; não devem ser tratados como implementação escalável para redes profundas.
- O experimento de pré-treinamento demonstra **transferência supervisionada por expansão de camadas**, mas a comparação de tempo total favorece o modelo já treinado.
- O curriculum learning usa grupos de ruído conhecido em dados sintéticos; em produção dificuldade dos exemplos precisa ser estimada.
- `--quick` usa pouquíssimas épocas, destinando-se somente a **teste de execução** e não à análise das curvas.

## Testes

```powershell
python -m unittest discover -s tests -v
```
